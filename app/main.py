from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from redis.asyncio import Redis

from app.config import load_policy_config
from app.core.analysis.service import PresidioAnalysisService
from app.guardrails import (
    GuardrailsAnalysisFailedError,
    GuardrailsBlockedError,
    GuardrailsError,
    GuardrailsNotFoundError,
    GuardrailsService,
    TextInput,
)
from app.model_assets import apply_model_env
from app.models.api import (
    ActionType,
    ApplyRequest,
    ApplyResponse,
    ApplyStreamRequest,
    ApplyStreamResponse,
    CapabilitiesResponse,
    ContentItem,
    Finding,
    FindingSpan,
    OutputItem,
    SessionFinalizeResponse,
    SessionState,
    StreamChunk,
    TimingInfo,
    UsageInfo,
)
from app.policy import PolicyResolver
from app.runtime.pytriton_embedded import EmbeddedPyTritonConfig, EmbeddedPyTritonManager
from app.settings import settings
from app.storage.redis_store import RedisMappingStore

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
if logger.isEnabledFor(logging.DEBUG):
    for noisy_logger in ("httpcore", "httpx", "filelock", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.INFO)

app = FastAPI(title="Guardrails Service", version="0.4.0")

_SUPPORTED_SOURCES = ["INPUT", "OUTPUT", "TOOL_INPUT", "TOOL_OUTPUT", "RETRIEVAL"]
_SUPPORTED_ACTIONS = ["NONE", "MASKED", "BLOCKED", "FLAGGED"]
_SUPPORTED_TRANSFORMS = ["reversible_mask"]
_SUPPORTED_TRANSFORM_MODES = ["DEIDENTIFY", "REIDENTIFY"]
_SUPPORTED_OUTPUT_SCOPES = ["INTERVENTIONS", "FULL"]
_SUPPORTED_TRACE_LEVELS = ["NONE", "BASIC", "FULL"]


def _to_inputs(items: list[ContentItem]) -> list[TextInput]:
    return [TextInput(id=item.id, text=item.text) for item in items]


def _items_payload(items: list[Any]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, dict):
            item_id = str(item.get("id", ""))
            text = str(item.get("text", ""))
        else:
            item_id = str(getattr(item, "id", ""))
            text = str(getattr(item, "text", ""))
        payload.append({"id": item_id, "text": text})
    return payload


def _debug_apply_log(
    *,
    stage: str,
    request_id: str | None,
    policy_id: str | None,
    source: str,
    input_items: list[Any],
    output_items: list[Any] | None = None,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "guardrails.apply stage=%s request_id=%s policy_id=%s source=%s input=%s output=%s",
        stage,
        request_id,
        policy_id,
        source,
        _items_payload(input_items),
        _items_payload(output_items or []),
    )


def _debug_stream_log(
    *,
    stage: str,
    request_id: str | None,
    policy_id: str | None,
    source: str,
    stream_id: str,
    final: bool,
    input_chunk: str,
    output_chunk: str,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "guardrails.apply-stream stage=%s request_id=%s policy_id=%s source=%s stream_id=%s final=%s input_chunk=%r output_chunk=%r",
        stage,
        request_id,
        policy_id,
        source,
        stream_id,
        final,
        input_chunk,
        output_chunk,
    )


def _score_to_severity(score: float) -> str:
    if score >= 0.95:
        return "critical"
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def _findings_from_result_items(
    *,
    items: list[Any],
    output_scope: str,
) -> list[Finding]:
    findings: list[Finding] = []
    for item in items:
        for detection in item.detections:
            metadata = detection.metadata or {}
            evidence: dict[str, Any] = {}
            if output_scope == "FULL":
                evidence = {
                    "item_id": item.id,
                    "entity_type": metadata.get("entity_type"),
                    "canonical_label": metadata.get("canonical_label"),
                    "detector": detection.detector,
                    "metadata": metadata,
                }
            findings.append(
                Finding(
                    check_id=str(detection.detector or "unknown"),
                    category=str(metadata.get("canonical_label") or detection.label.lower()),
                    severity=_score_to_severity(float(detection.score)),
                    confidence=float(detection.score),
                    spans=[
                        FindingSpan(
                            start=int(detection.start),
                            end=int(detection.end),
                            snippet=str(detection.text),
                            label=str(detection.label),
                        )
                    ],
                    evidence=evidence,
                )
            )
    return findings


def _missing_session_finding(session_id: str) -> Finding:
    return Finding(
        check_id="reversible_mask_session",
        category="session",
        severity="high",
        confidence=1.0,
        spans=[],
        evidence={
            "reason": "missing_or_expired_session",
            "session_id": session_id,
        },
    )


def _analysis_failed_finding(detector_errors: dict[str, str], *, output_scope: str) -> Finding:
    evidence: dict[str, Any] = {"reason": "analysis_failed"}
    if output_scope == "FULL":
        evidence["detector_errors"] = dict(detector_errors)
    return Finding(
        check_id="analysis",
        category="system",
        severity="critical",
        confidence=1.0,
        spans=[],
        evidence=evidence,
    )

def _models_not_ready_finding(*, output_scope: str) -> Finding:
    evidence: dict[str, Any] = {"reason": "models_not_ready"}
    if output_scope == "FULL":
        evidence["load_error"] = str(getattr(app.state, "models_load_error", None) or "")
    return Finding(
        check_id="models",
        category="system",
        severity="critical",
        confidence=1.0,
        spans=[],
        evidence=evidence,
    )


def _usage_from_io(input_content: list[ContentItem], output_items: list[OutputItem]) -> UsageInfo:
    return UsageInfo(
        input_items=len(input_content),
        input_chars=sum(len(item.text) for item in input_content),
        output_items=len(output_items),
        output_chars=sum(len(item.text) for item in output_items),
    )


def _timings_from_result_items(*, started_at: float, items: list[Any]) -> TimingInfo:
    detector_timing_ms: dict[str, float] = {}
    for item in items:
        diagnostics = item.diagnostics or {}
        detector_map = diagnostics.get("detector_timing_ms", {})
        if isinstance(detector_map, dict):
            for key, value in detector_map.items():
                detector_timing_ms[str(key)] = detector_timing_ms.get(str(key), 0.0) + float(value)
    return TimingInfo(
        total_ms=round((perf_counter() - started_at) * 1000.0, 3),
        detector_timing_ms={key: round(value, 3) for key, value in detector_timing_ms.items()},
    )


def _timings_from_elapsed_ms(*, started_at: float) -> TimingInfo:
    return TimingInfo(total_ms=round((perf_counter() - started_at) * 1000.0, 3), detector_timing_ms={})


def _session_state(*, session_id: str, ttl_seconds: int) -> SessionState:
    expires_at = (datetime.now(tz=UTC) + timedelta(seconds=ttl_seconds)).isoformat()
    return SessionState(id=session_id, ttl_seconds=ttl_seconds, expires_at=expires_at)


def _resolve_policy_name(policy_id: str | None) -> str:
    resolved_name, _ = app.state.policy_resolver.resolve_policy(policy_name=policy_id)
    return resolved_name


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _build_embedded_pytriton_manager() -> EmbeddedPyTritonManager:
    return EmbeddedPyTritonManager(
        EmbeddedPyTritonConfig(
            pytriton_url=settings.pytriton_url,
            enable_gliner=settings.enable_gliner,
            gliner_model_ref=_env("GR_PYTRITON_GLINER_MODEL_REF", "urchade/gliner_multi-v2.1"),
            token_model_ref=_env("GR_PYTRITON_TOKEN_MODEL_REF", "scanpatch/pii-ner-nemotron"),
            model_dir=settings.model_dir,
            offline_mode=settings.offline_mode,
            device=_env("GR_PYTRITON_DEVICE", "cuda"),
            max_batch_size=int(_env("GR_PYTRITON_MAX_BATCH_SIZE", "32")),
            enable_nemotron=settings.enable_nemotron,
            grpc_port=int(_env("GR_PYTRITON_GRPC_PORT", "8001")),
            metrics_port=int(_env("GR_PYTRITON_METRICS_PORT", "8002")),
            readiness_timeout_s=settings.pytriton_init_timeout_s,
        )
    )


def _load_runtime() -> None:
    config = load_policy_config(settings.policy_path)
    resolver = PolicyResolver(config)
    analysis_service = PresidioAnalysisService(config)
    app.state.policy_config = config
    app.state.policy_resolver = resolver
    app.state.analysis_service = analysis_service
    app.state.guardrails = GuardrailsService(
        policy_resolver=resolver,
        analysis_service=analysis_service,
        mapping_store=app.state.mapping_store,
    )
    logger.info(
        "policy loaded from %s, profiles=%s, recognizers=%s",
        settings.policy_path,
        ",".join(sorted(config.analyzer_profiles.keys())),
        ",".join(sorted(config.recognizer_definitions.keys())),
    )

async def _initialize_models_in_background() -> None:
    # Run model startup/readiness in the background. The request path will fail
    # closed (BLOCKED, outputs=[]) while models_ready is false.
    try:
        app.state.models_load_error = None
        if settings.runtime_mode == "cuda":
            if not bool(settings.enable_gliner) and not bool(settings.enable_nemotron):
                # No ML models enabled: skip Triton entirely.
                _load_runtime()
                config = app.state.policy_resolver.config
                readiness_errors = app.state.analysis_service.ensure_profile_runtimes_ready(
                    profile_names=sorted(config.analyzer_profiles.keys()),
                    timeout_s=None,
                )
                if readiness_errors:
                    raise RuntimeError(f"model runtime readiness check failed: {readiness_errors}")
                app.state.models_ready = True
                app.state.models_load_error = None
                return
            manager = getattr(app.state, "embedded_pytriton_manager", None)
            if manager is None:
                manager = _build_embedded_pytriton_manager()
                app.state.embedded_pytriton_manager = manager
            await asyncio.to_thread(manager.start)
            settings.pytriton_url = manager.client_url

        _load_runtime()
        config = app.state.policy_resolver.config
        readiness_errors = app.state.analysis_service.ensure_profile_runtimes_ready(
            profile_names=sorted(config.analyzer_profiles.keys()),
            timeout_s=None,
        )
        if readiness_errors:
            raise RuntimeError(f"model runtime readiness check failed: {readiness_errors}")
        app.state.models_ready = True
        app.state.models_load_error = None
    except Exception as exc:
        app.state.models_ready = False
        app.state.models_load_error = str(exc)
        logger.exception("model initialization failed: %s", exc)


def _schedule_model_init() -> None:
    app.state.models_ready = False
    task = getattr(app.state, "models_init_task", None)
    if task is not None and not task.done():
        task.cancel()
    app.state.models_init_task = asyncio.create_task(_initialize_models_in_background())


@app.on_event("startup")
async def startup() -> None:
    apply_model_env(model_dir=settings.model_dir, offline_mode=settings.offline_mode)
    app.state.models_ready = False
    app.state.models_load_error = None
    app.state.embedded_pytriton_manager = None

    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=False)
    app.state.mapping_store = RedisMappingStore(app.state.redis)

    # Load policy immediately (fast), then initialize models/readiness in background.
    try:
        _load_runtime()
    except Exception as exc:
        app.state.models_load_error = str(exc)
        logger.exception("policy load failed: %s", exc)
    _schedule_model_init()


@app.on_event("shutdown")
async def shutdown() -> None:
    task = getattr(app.state, "models_init_task", None)
    if task is not None and not task.done():
        task.cancel()
    manager = getattr(app.state, "embedded_pytriton_manager", None)
    if manager is not None:
        manager.stop()
    redis_client = getattr(app.state, "redis", None)
    if redis_client is not None:
        await redis_client.aclose()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    try:
        if not bool(getattr(app.state, "models_ready", False)):
            load_error = str(getattr(app.state, "models_load_error", None) or "")
            raise RuntimeError(load_error or "model runtimes are still loading")
        pong = await app.state.redis.ping()
        if not pong:
            raise RuntimeError("redis ping returned false")
        if settings.runtime_mode == "cuda":
            manager = getattr(app.state, "embedded_pytriton_manager", None)
            if manager is None:
                raise RuntimeError("embedded pytriton manager is not initialized")
            if not manager.is_ready():
                raise RuntimeError(manager.last_error() or "embedded pytriton is not ready")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"not ready: {exc}") from exc
    return {"status": "ready"}


@app.post("/admin/reload")
async def reload_policy() -> dict[str, Any]:
    _load_runtime()
    _schedule_model_init()
    config = app.state.policy_resolver.config
    return {
        "status": "reloaded",
        "policy_path": settings.policy_path,
        "policies": app.state.policy_resolver.list_policies(),
        "analyzer_profiles": sorted(config.analyzer_profiles.keys()),
        "recognizers": sorted(config.recognizer_definitions.keys()),
    }


@app.get("/v1/guardrails/capabilities", response_model=CapabilitiesResponse)
async def capabilities_endpoint() -> CapabilitiesResponse:
    config = app.state.policy_resolver.config
    return CapabilitiesResponse(
        service=settings.service_name,
        api_version="v1",
        sources=_SUPPORTED_SOURCES,
        actions=_SUPPORTED_ACTIONS,
        transforms=_SUPPORTED_TRANSFORMS,
        transform_modes=_SUPPORTED_TRANSFORM_MODES,
        output_scopes=_SUPPORTED_OUTPUT_SCOPES,
        trace_levels=_SUPPORTED_TRACE_LEVELS,
        policies=app.state.policy_resolver.list_policies(),
        checks=sorted(config.recognizer_definitions.keys()),
        runtime_mode=settings.runtime_mode,
    )


@app.post("/v1/guardrails/apply", response_model=ApplyResponse)
async def apply_endpoint(request: ApplyRequest) -> ApplyResponse:
    started_at = perf_counter()
    try:
        policy_name, policy = app.state.policy_resolver.resolve_policy(policy_name=request.policy_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    transforms = request.transforms
    if len(transforms) > 1:
        raise HTTPException(status_code=400, detail="only one transform is supported per request")

    content_inputs = _to_inputs(request.content)
    output_items: list[OutputItem] = [OutputItem(id=item.id, text=item.text) for item in request.content]

    if not transforms:
        if not bool(getattr(app.state, "models_ready", False)):
            return ApplyResponse(
                action="BLOCKED",
                source=request.source,
                policy_id=policy_name,
                policy_version=request.policy_version,
                outputs=[],
                findings=[_models_not_ready_finding(output_scope=request.output_scope)],
                session=None,
                usage=UsageInfo(
                    input_items=len(request.content),
                    input_chars=sum(len(item.text) for item in request.content),
                    output_items=0,
                    output_chars=0,
                ),
                timings=_timings_from_elapsed_ms(started_at=started_at),
            )
        try:
            result = await app.state.guardrails.detect_items(items=content_inputs, policy_name=policy_name)
            findings = _findings_from_result_items(items=result.items, output_scope=request.output_scope)
            action: ActionType = "FLAGGED" if findings else "NONE"
        except GuardrailsAnalysisFailedError as exc:
            findings = [_analysis_failed_finding(exc.detector_errors, output_scope=request.output_scope)]
            _debug_apply_log(
                stage="detect_analysis_failed",
                request_id=request.request_id,
                policy_id=policy_name,
                source=request.source,
                input_items=request.content,
                output_items=[],
            )
            return ApplyResponse(
                action="BLOCKED",
                source=request.source,
                policy_id=policy_name,
                policy_version=request.policy_version,
                outputs=[],
                findings=findings,
                session=None,
                usage=UsageInfo(
                    input_items=len(request.content),
                    input_chars=sum(len(item.text) for item in request.content),
                    output_items=0,
                    output_chars=0,
                ),
                timings=_timings_from_elapsed_ms(started_at=started_at),
            )
        _debug_apply_log(
            stage="detect",
            request_id=request.request_id,
            policy_id=policy_name,
            source=request.source,
            input_items=request.content,
            output_items=output_items,
        )
        return ApplyResponse(
            action=action,
            source=request.source,
            policy_id=result.policy_name,
            policy_version=request.policy_version,
            outputs=output_items,
            findings=findings,
            session=None,
            usage=_usage_from_io(request.content, output_items),
            timings=_timings_from_result_items(started_at=started_at, items=result.items),
        )

    transform = transforms[0]

    if transform.mode == "DEIDENTIFY":
        if not bool(getattr(app.state, "models_ready", False)):
            return ApplyResponse(
                action="BLOCKED",
                source=request.source,
                policy_id=policy_name,
                policy_version=request.policy_version,
                outputs=[],
                findings=[_models_not_ready_finding(output_scope=request.output_scope)],
                session=None,
                usage=UsageInfo(
                    input_items=len(request.content),
                    input_chars=sum(len(item.text) for item in request.content),
                    output_items=0,
                    output_chars=0,
                ),
                timings=_timings_from_elapsed_ms(started_at=started_at),
            )
        session_id = transform.session.id or f"sess_{uuid4().hex}"
        session_ttl_seconds = int(transform.session.ttl_seconds or policy.storage_ttl_seconds)
        try:
            masked = await app.state.guardrails.mask_items(
                request_id=session_id,
                items=content_inputs,
                policy_name=policy_name,
                store_context=True,
                storage_ttl_seconds=session_ttl_seconds,
            )
        except GuardrailsAnalysisFailedError as exc:
            findings = [_analysis_failed_finding(exc.detector_errors, output_scope=request.output_scope)]
            _debug_apply_log(
                stage="deidentify_analysis_failed",
                request_id=request.request_id or session_id,
                policy_id=policy_name,
                source=request.source,
                input_items=request.content,
                output_items=[],
            )
            return ApplyResponse(
                action="BLOCKED",
                source=request.source,
                policy_id=policy_name,
                policy_version=request.policy_version,
                outputs=[],
                findings=findings,
                session=None,
                usage=UsageInfo(
                    input_items=len(request.content),
                    input_chars=sum(len(item.text) for item in request.content),
                    output_items=0,
                    output_chars=0,
                ),
                timings=_timings_from_elapsed_ms(started_at=started_at),
            )
        except GuardrailsBlockedError:
            try:
                detected = await app.state.guardrails.detect_items(items=content_inputs, policy_name=policy_name)
                findings = _findings_from_result_items(items=detected.items, output_scope=request.output_scope)
            except GuardrailsAnalysisFailedError as exc:
                findings = [_analysis_failed_finding(exc.detector_errors, output_scope=request.output_scope)]
            _debug_apply_log(
                stage="deidentify_blocked",
                request_id=request.request_id or session_id,
                policy_id=policy_name,
                source=request.source,
                input_items=request.content,
                output_items=[],
            )
            return ApplyResponse(
                action="BLOCKED",
                source=request.source,
                policy_id=policy_name,
                policy_version=request.policy_version,
                outputs=[],
                findings=findings,
                session=None,
                usage=UsageInfo(
                    input_items=len(request.content),
                    input_chars=sum(len(item.text) for item in request.content),
                    output_items=0,
                    output_chars=0,
                ),
                timings=_timings_from_elapsed_ms(started_at=started_at),
            )
        except GuardrailsError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        findings = _findings_from_result_items(items=masked.items, output_scope=request.output_scope)
        output_items = [OutputItem(id=item.id, text=item.text) for item in masked.items]
        _debug_apply_log(
            stage="deidentify",
            request_id=request.request_id or session_id,
            policy_id=masked.policy_name,
            source=request.source,
            input_items=request.content,
            output_items=output_items,
        )
        action = "MASKED" if masked.placeholders_count > 0 else "NONE"
        return ApplyResponse(
            action=action,
            source=request.source,
            policy_id=masked.policy_name,
            policy_version=request.policy_version,
            outputs=output_items,
            findings=findings,
            session=_session_state(session_id=session_id, ttl_seconds=session_ttl_seconds),
            usage=_usage_from_io(request.content, output_items),
            timings=_timings_from_result_items(started_at=started_at, items=masked.items),
        )

    if transform.mode == "REIDENTIFY":
        session_id = transform.session.id
        if not session_id:
            raise HTTPException(status_code=400, detail="REIDENTIFY requires transforms[0].session.id")
        allow_missing_context = (
            transform.session.allow_missing_context
            if transform.session.allow_missing_context is not None
            else settings.allow_missing_reidentify_session
        )
        try:
            unmasked = await app.state.guardrails.unmask_items(
                request_id=session_id,
                items=content_inputs,
                delete_context=False,
                allow_missing_context=allow_missing_context,
            )
        except GuardrailsNotFoundError:
            findings = [_missing_session_finding(session_id)]
            _debug_apply_log(
                stage="reidentify_missing_session_blocked",
                request_id=request.request_id or session_id,
                policy_id=policy_name,
                source=request.source,
                input_items=request.content,
                output_items=[],
            )
            return ApplyResponse(
                action="BLOCKED",
                source=request.source,
                policy_id=policy_name,
                policy_version=request.policy_version,
                outputs=[],
                findings=findings,
                session=None,
                usage=UsageInfo(
                    input_items=len(request.content),
                    input_chars=sum(len(item.text) for item in request.content),
                    output_items=0,
                    output_chars=0,
                ),
                timings=_timings_from_elapsed_ms(started_at=started_at),
            )

        if not unmasked.context_found:
            findings = [_missing_session_finding(session_id)]
            _debug_apply_log(
                stage="reidentify_missing_session_flagged",
                request_id=request.request_id or session_id,
                policy_id=policy_name,
                source=request.source,
                input_items=request.content,
                output_items=output_items,
            )
            return ApplyResponse(
                action="FLAGGED",
                source=request.source,
                policy_id=policy_name,
                policy_version=request.policy_version,
                outputs=output_items,
                findings=findings,
                session=None,
                usage=_usage_from_io(request.content, output_items),
                timings=_timings_from_elapsed_ms(started_at=started_at),
            )

        output_items = [OutputItem(id=item.id, text=item.text) for item in unmasked.items]
        _debug_apply_log(
            stage="reidentify",
            request_id=request.request_id or session_id,
            policy_id=policy_name,
            source=request.source,
            input_items=request.content,
            output_items=output_items,
        )
        return ApplyResponse(
            action="MASKED" if unmasked.replacements > 0 else "NONE",
            source=request.source,
            policy_id=policy_name,
            policy_version=request.policy_version,
            outputs=output_items,
            findings=[],
            session=None,
            usage=_usage_from_io(request.content, output_items),
            timings=_timings_from_elapsed_ms(started_at=started_at),
        )

    raise HTTPException(status_code=400, detail=f"unsupported transform mode: {transform.mode}")


@app.post("/v1/guardrails/apply-stream", response_model=ApplyStreamResponse)
async def apply_stream_endpoint(request: ApplyStreamRequest) -> ApplyStreamResponse:
    started_at = perf_counter()
    try:
        policy_name = _resolve_policy_name(request.policy_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    transforms = request.transforms
    if len(transforms) != 1:
        raise HTTPException(status_code=400, detail="apply-stream requires exactly one transform")
    transform = transforms[0]
    if transform.mode != "REIDENTIFY":
        raise HTTPException(status_code=400, detail="apply-stream supports only REIDENTIFY transform")

    session_id = transform.session.id
    if not session_id:
        raise HTTPException(status_code=400, detail="REIDENTIFY requires transforms[0].session.id")

    allow_missing_context = (
        transform.session.allow_missing_context
        if transform.session.allow_missing_context is not None
        else settings.allow_missing_reidentify_session
    )

    try:
        result = await app.state.guardrails.unmask_stream_chunk(
            request_id=session_id,
            stream_id=request.stream.id,
            chunk=request.stream.chunk,
            final=request.stream.final,
            delete_context=False,
            allow_missing_context=allow_missing_context,
        )
    except GuardrailsNotFoundError:
        findings = [_missing_session_finding(session_id)]
        _debug_stream_log(
            stage="reidentify_stream_missing_session_blocked",
            request_id=request.request_id or session_id,
            policy_id=policy_name,
            source=request.source,
            stream_id=request.stream.id,
            final=request.stream.final,
            input_chunk=request.stream.chunk,
            output_chunk="",
        )
        return ApplyStreamResponse(
            action="BLOCKED",
            source=request.source,
            policy_id=policy_name,
            policy_version=request.policy_version,
            stream=StreamChunk(id=request.stream.id, chunk=request.stream.chunk, final=request.stream.final),
            output_chunk="",
            replacements=0,
            buffered_chars=0,
            findings=findings,
            session=None,
            usage=UsageInfo(input_items=1, input_chars=len(request.stream.chunk), output_items=0, output_chars=0),
            timings=_timings_from_elapsed_ms(started_at=started_at),
        )
    except GuardrailsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    action: ActionType = "NONE"
    findings: list[Finding] = []
    if not result.context_found:
        findings = [_missing_session_finding(session_id)]
        action = "FLAGGED"
    elif result.replacements > 0:
        action = "MASKED"

    output_chunk = result.output
    _debug_stream_log(
        stage="reidentify_stream",
        request_id=request.request_id or session_id,
        policy_id=policy_name,
        source=request.source,
        stream_id=request.stream.id,
        final=request.stream.final,
        input_chunk=request.stream.chunk,
        output_chunk=output_chunk,
    )
    output_chars = len(output_chunk)
    output_items = 1 if output_chunk else 0
    return ApplyStreamResponse(
        action=action,
        source=request.source,
        policy_id=policy_name,
        policy_version=request.policy_version,
        stream=StreamChunk(id=request.stream.id, chunk=request.stream.chunk, final=request.stream.final),
        output_chunk=output_chunk,
        replacements=result.replacements,
        buffered_chars=result.buffered_chars,
        findings=findings,
        session=None,
        usage=UsageInfo(
            input_items=1,
            input_chars=len(request.stream.chunk),
            output_items=output_items,
            output_chars=output_chars,
        ),
        timings=_timings_from_elapsed_ms(started_at=started_at),
    )


@app.post("/v1/guardrails/sessions/{session_id}/finalize", response_model=SessionFinalizeResponse)
async def finalize_session_endpoint(session_id: str) -> SessionFinalizeResponse:
    deleted = await app.state.guardrails.finalize_request(request_id=session_id)
    return SessionFinalizeResponse(session_id=session_id, context_deleted=deleted)
