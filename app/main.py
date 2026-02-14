from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from redis.asyncio import Redis

from app.config import load_policy_config
from app.detectors.factory import build_registry
from app.guardrails import (
    GuardrailsBlockedError,
    GuardrailsError,
    GuardrailsNotFoundError,
    GuardrailsService,
    TextInput,
)
from app.models.api import (
    DetectRequest,
    DetectResponse,
    DetectResultItem,
    DetectionItem,
    FinalizeRequest,
    FinalizeResponse,
    MaskRequest,
    MaskResponse,
    MaskResultItem,
    PolicyListResponse,
    StreamUnmaskRequest,
    StreamUnmaskResponse,
    UnmaskRequest,
    UnmaskResponse,
    UnmaskResultItem,
)
from app.policy import PolicyResolver
from app.settings import settings
from app.storage.redis_store import RedisMappingStore

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Guardrails Service", version="0.2.0")


def _to_inputs(items) -> list[TextInput]:
    return [TextInput(id=item.id, text=item.text) for item in items]


def _to_detection_items(detections) -> list[DetectionItem]:
    return [
        DetectionItem(
            start=item.start,
            end=item.end,
            label=item.label,
            score=item.score,
            detector=item.detector,
            snippet=item.text,
        )
        for item in detections
    ]


def _load_runtime() -> None:
    config = load_policy_config(settings.policy_path)
    resolver = PolicyResolver(config)
    registry = build_registry(config.detector_definitions)
    app.state.policy_resolver = resolver
    app.state.detector_registry = registry
    app.state.guardrails = GuardrailsService(
        policy_resolver=resolver,
        detector_registry=registry,
        mapping_store=app.state.mapping_store,
    )
    logger.info("policy loaded from %s, active detectors=%s", settings.policy_path, ",".join(registry.names()))


@app.on_event("startup")
async def startup() -> None:
    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=False)
    app.state.mapping_store = RedisMappingStore(app.state.redis)
    _load_runtime()


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.redis.aclose()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/admin/reload")
async def reload_policy() -> dict[str, Any]:
    _load_runtime()
    return {
        "status": "reloaded",
        "policy_path": settings.policy_path,
        "detectors": app.state.detector_registry.names(),
        "policies": app.state.policy_resolver.list_policies(),
    }


@app.get("/v1/guardrails/policies", response_model=PolicyListResponse)
async def list_policies() -> PolicyListResponse:
    return PolicyListResponse(
        default_policy=app.state.policy_resolver.config.default_policy,
        policies=app.state.policy_resolver.list_policies(),
    )


@app.post("/v1/guardrails/detect", response_model=DetectResponse)
async def detect_endpoint(request: DetectRequest) -> DetectResponse:
    try:
        result = await app.state.guardrails.detect_items(
            items=_to_inputs(request.items),
            policy_name=request.policy_name,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DetectResponse(
        policy_name=result.policy_name,
        mode=result.mode,
        findings_count=result.findings_count,
        items=[
            DetectResultItem(id=item.id, detections=_to_detection_items(item.detections))
            for item in result.items
        ],
    )


@app.post("/v1/guardrails/mask", response_model=MaskResponse)
async def mask_endpoint(request: MaskRequest) -> MaskResponse:
    try:
        result = await app.state.guardrails.mask_items(
            request_id=request.request_id,
            items=_to_inputs(request.items),
            policy_name=request.policy_name,
            store_context=request.store_context,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GuardrailsBlockedError as exc:
        raise HTTPException(
            status_code=403,
            detail={"message": str(exc), "findings_count": exc.findings_count},
        ) from exc
    except GuardrailsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return MaskResponse(
        request_id=result.request_id,
        policy_name=result.policy_name,
        mode=result.mode,
        findings_count=result.findings_count,
        placeholders_count=result.placeholders_count,
        context_stored=result.context_stored,
        items=[
            MaskResultItem(id=item.id, text=item.text, detections=_to_detection_items(item.detections))
            for item in result.items
        ],
    )


@app.post("/v1/guardrails/unmask", response_model=UnmaskResponse)
async def unmask_endpoint(request: UnmaskRequest) -> UnmaskResponse:
    try:
        result = await app.state.guardrails.unmask_items(
            request_id=request.request_id,
            items=_to_inputs(request.items),
            delete_context=request.delete_context,
            allow_missing_context=request.allow_missing_context,
        )
    except GuardrailsNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return UnmaskResponse(
        request_id=result.request_id,
        context_found=result.context_found,
        replacements=result.replacements,
        context_deleted=result.context_deleted,
        items=[
            UnmaskResultItem(id=item.id, text=item.text, replacements=item.replacements)
            for item in result.items
        ],
    )


@app.post("/v1/guardrails/unmask-stream", response_model=StreamUnmaskResponse)
async def unmask_stream_endpoint(request: StreamUnmaskRequest) -> StreamUnmaskResponse:
    try:
        result = await app.state.guardrails.unmask_stream_chunk(
            request_id=request.request_id,
            stream_id=request.stream_id,
            chunk=request.chunk,
            final=request.final,
            delete_context=request.delete_context,
            allow_missing_context=request.allow_missing_context,
        )
    except GuardrailsNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GuardrailsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamUnmaskResponse(
        request_id=result.request_id,
        stream_id=result.stream_id,
        context_found=result.context_found,
        output=result.output,
        replacements=result.replacements,
        buffered_chars=result.buffered_chars,
        final=result.final,
        context_deleted=result.context_deleted,
    )


@app.post("/v1/guardrails/finalize", response_model=FinalizeResponse)
async def finalize_endpoint(request: FinalizeRequest) -> FinalizeResponse:
    context_deleted = await app.state.guardrails.finalize_request(request_id=request.request_id)
    return FinalizeResponse(request_id=request.request_id, context_deleted=context_deleted)
