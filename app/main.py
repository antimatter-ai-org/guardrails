from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from redis.asyncio import Redis

from app.config import load_policy_config
from app.core.analysis.service import PresidioAnalysisService
from app.guardrails import (
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
from app.settings import settings
from app.storage.redis_store import RedisMappingStore

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Guardrails Service", version="0.4.0")

_SUPPORTED_SOURCES = ["INPUT", "OUTPUT", "TOOL_INPUT", "TOOL_OUTPUT", "RETRIEVAL"]
_SUPPORTED_ACTIONS = ["NONE", "MASKED", "BLOCKED", "FLAGGED"]
_SUPPORTED_TRANSFORMS = ["reversible_mask"]
_SUPPORTED_TRANSFORM_MODES = ["DEIDENTIFY", "REIDENTIFY"]
_SUPPORTED_OUTPUT_SCOPES = ["INTERVENTIONS", "FULL"]
_SUPPORTED_TRACE_LEVELS = ["NONE", "BASIC", "FULL"]


def _to_inputs(items: list[ContentItem]) -> list[TextInput]:
    return [TextInput(id=item.id, text=item.text) for item in items]


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


@app.on_event("startup")
async def startup() -> None:
    apply_model_env(model_dir=settings.model_dir, offline_mode=settings.offline_mode)
    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=False)
    app.state.mapping_store = RedisMappingStore(app.state.redis)
    _load_runtime()


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.redis.aclose()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    try:
        pong = await app.state.redis.ping()
        if not pong:
            raise RuntimeError("redis ping returned false")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"not ready: {exc}") from exc
    return {"status": "ready"}


@app.post("/admin/reload")
async def reload_policy() -> dict[str, Any]:
    _load_runtime()
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
        result = await app.state.guardrails.detect_items(items=content_inputs, policy_name=policy_name)
        findings = _findings_from_result_items(items=result.items, output_scope=request.output_scope)
        action: ActionType = "FLAGGED" if findings else "NONE"
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
        except GuardrailsBlockedError:
            detected = await app.state.guardrails.detect_items(items=content_inputs, policy_name=policy_name)
            findings = _findings_from_result_items(items=detected.items, output_scope=request.output_scope)
            return ApplyResponse(
                action="BLOCKED",
                source=request.source,
                policy_id=detected.policy_name,
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
                timings=_timings_from_result_items(started_at=started_at, items=detected.items),
            )
        except GuardrailsError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        findings = _findings_from_result_items(items=masked.items, output_scope=request.output_scope)
        output_items = [OutputItem(id=item.id, text=item.text) for item in masked.items]
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
