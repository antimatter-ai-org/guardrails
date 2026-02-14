from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from app.config import load_policy_config
from app.detectors.factory import build_registry
from app.guardrails import GuardrailsError, GuardrailsService
from app.models.openai import ChatCompletionRequest, GuardRequest, GuardResponse
from app.policy import PolicyResolver
from app.proxy.upstream import UpstreamClient
from app.settings import settings
from app.storage.redis_store import RedisMappingStore

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Guardrails Service", version="0.1.0")


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
    app.state.upstream_client = UpstreamClient(timeout_seconds=settings.http_timeout_seconds)
    _load_runtime()


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.upstream_client.close()
    await app.state.redis.close()


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
    }


@app.post("/v1/guardrails/mask")
async def mask_endpoint(request: GuardRequest) -> dict[str, Any]:
    payload = request.payload
    model_name = str(payload.get("model", ""))
    if not model_name:
        raise HTTPException(status_code=400, detail="payload.model is required")

    masked_payload, policy_name, context = await app.state.guardrails.mask_request(
        request_id=request.request_id,
        payload=payload,
        model_name=model_name,
    )
    return {
        "request_id": request.request_id,
        "policy": policy_name,
        "payload": masked_payload,
        "masked_values": len(context.get("placeholders", {})),
    }


@app.post("/v1/guardrails/unmask")
async def unmask_endpoint(request: GuardResponse) -> dict[str, Any]:
    payload = await app.state.guardrails.unmask_response(
        request_id=request.request_id,
        payload=request.payload,
    )
    return {
        "request_id": request.request_id,
        "payload": payload,
    }


@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request) -> Response:
    raw_payload = await request.json()
    validated = ChatCompletionRequest.model_validate(raw_payload)
    if validated.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported in MVP")

    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    policy_name_override = request.headers.get("x-guardrails-policy")
    upstream_base_url = request.headers.get("x-upstream-base-url", settings.default_upstream_base_url)

    try:
        masked_payload, policy_name, context = await app.state.guardrails.mask_request(
            request_id=request_id,
            payload=raw_payload,
            model_name=validated.model,
            policy_name=policy_name_override,
        )
    except GuardrailsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    forward_headers: dict[str, str] = {}
    if "authorization" in request.headers:
        forward_headers["authorization"] = request.headers["authorization"]

    status_code, upstream_payload, upstream_headers = await app.state.upstream_client.chat_completions(
        base_url=upstream_base_url,
        payload=masked_payload,
        headers=forward_headers,
    )

    if status_code >= 400:
        return JSONResponse(status_code=status_code, content=upstream_payload)

    unmasked_payload = await app.state.guardrails.unmask_response(request_id=request_id, payload=upstream_payload)

    response = JSONResponse(status_code=status_code, content=unmasked_payload)
    response.headers["x-guardrails-request-id"] = request_id
    response.headers["x-guardrails-policy"] = policy_name
    response.headers["x-guardrails-masked-values"] = str(len(context.get("placeholders", {})))
    if upstream_headers.get("x-request-id"):
        response.headers["x-upstream-request-id"] = upstream_headers["x-request-id"]
    return response
