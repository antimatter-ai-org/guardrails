from __future__ import annotations

from typing import Any

import pytest

from app.config import DetectorDefinition, PolicyConfig, PolicyDefinition, RoutingConfig
from app.detectors.factory import build_registry
from app.guardrails import GuardrailsService
from app.policy import PolicyResolver


class InMemoryStore:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}

    async def save(self, request_id: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        self.data[request_id] = payload

    async def load(self, request_id: str) -> dict[str, Any] | None:
        return self.data.get(request_id)

    async def delete(self, request_id: str) -> None:
        self.data.pop(request_id, None)


@pytest.mark.asyncio
async def test_mask_and_unmask_request_response_flow() -> None:
    config = PolicyConfig(
        default_policy="external_default",
        routing=RoutingConfig(external_model_patterns=[r"^gpt-"], default_destination="external"),
        policies={
            "external_default": PolicyDefinition(
                mode="mask",
                detectors=["email_regex"],
                min_score=0.5,
                storage_ttl_seconds=300,
                placeholder_prefix="GR",
            )
        },
        detector_definitions={
            "email_regex": DetectorDefinition(
                type="regex",
                enabled=True,
                params={
                    "patterns": [
                            {
                                "name": "email",
                                "label": "EMAIL",
                                "pattern": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                                "score": 0.98,
                            }
                        ]
                    },
            )
        },
    )

    registry = build_registry(config.detector_definitions)
    resolver = PolicyResolver(config)
    store = InMemoryStore()
    service = GuardrailsService(policy_resolver=resolver, detector_registry=registry, mapping_store=store)  # type: ignore[arg-type]

    request_payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Почта ivan@example.com"}],
    }

    masked_payload, policy_name, context = await service.mask_request(
        request_id="req-1",
        payload=request_payload,
        model_name="gpt-4o-mini",
    )

    assert policy_name == "external_default"
    assert context["placeholders"]
    assert "ivan@example.com" not in masked_payload["messages"][0]["content"]

    response_payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"Получил {next(iter(context['placeholders']))}"},
                "finish_reason": "stop",
            }
        ],
    }

    unmasked = await service.unmask_response("req-1", response_payload)
    assert "ivan@example.com" in unmasked["choices"][0]["message"]["content"]
