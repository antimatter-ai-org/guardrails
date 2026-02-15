from __future__ import annotations

from typing import Any

import pytest

from app.config import AnalysisConfig, AnalyzerProfile, PolicyConfig, PolicyDefinition, RecognizerDefinition
from app.core.analysis.service import PresidioAnalysisService
from app.guardrails import GuardrailsBlockedError, GuardrailsService, TextInput
from app.policy import PolicyResolver


class InMemoryStore:
    def __init__(self) -> None:
        self.mappings: dict[str, dict[str, Any]] = {}
        self.streams: dict[tuple[str, str], dict[str, Any]] = {}

    async def save_mapping(self, request_id: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        self.mappings[request_id] = payload

    async def load_mapping(self, request_id: str) -> dict[str, Any] | None:
        return self.mappings.get(request_id)

    async def delete_mapping(self, request_id: str) -> None:
        self.mappings.pop(request_id, None)

    async def save_stream_state(self, request_id: str, stream_id: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        self.streams[(request_id, stream_id)] = payload

    async def load_stream_state(self, request_id: str, stream_id: str) -> dict[str, Any] | None:
        return self.streams.get((request_id, stream_id))

    async def delete_stream_state(self, request_id: str, stream_id: str) -> None:
        self.streams.pop((request_id, stream_id), None)

    async def delete_all_stream_states(self, request_id: str) -> int:
        keys = [key for key in self.streams if key[0] == request_id]
        for key in keys:
            self.streams.pop(key, None)
        return len(keys)

    async def delete_all_for_request(self, request_id: str) -> None:
        await self.delete_mapping(request_id)
        await self.delete_all_stream_states(request_id)


def _make_service() -> GuardrailsService:
    config = PolicyConfig(
        default_policy="external_default",
        policies={
            "external_default": PolicyDefinition(
                mode="mask",
                analyzer_profile="regex_profile",
                min_score=0.5,
                storage_ttl_seconds=300,
                placeholder_prefix="GR",
            ),
            "strict_block": PolicyDefinition(
                mode="block",
                analyzer_profile="regex_profile",
                min_score=0.5,
                storage_ttl_seconds=300,
                placeholder_prefix="GR",
            ),
            "onprem_passthrough": PolicyDefinition(
                mode="passthrough",
                analyzer_profile="empty_profile",
                min_score=1.0,
                storage_ttl_seconds=300,
                placeholder_prefix="GR",
            ),
        },
        analyzer_profiles={
            "regex_profile": AnalyzerProfile(
                analysis=AnalysisConfig(recognizers=["email_regex"]),
            ),
            "empty_profile": AnalyzerProfile(
                analysis=AnalysisConfig(recognizers=[]),
            ),
        },
        recognizer_definitions={
            "email_regex": RecognizerDefinition(
                type="regex",
                enabled=True,
                params={
                    "patterns": [
                        {
                            "name": "email",
                            "label": "EMAIL_ADDRESS",
                            "pattern": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                            "score": 0.98,
                        }
                    ]
                },
            )
        },
    )

    resolver = PolicyResolver(config)
    store = InMemoryStore()
    analysis = PresidioAnalysisService(config)
    return GuardrailsService(policy_resolver=resolver, analysis_service=analysis, mapping_store=store)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mask_and_unmask_batch_flow() -> None:
    service = _make_service()

    masked = await service.mask_items(
        request_id="req-1",
        policy_name="external_default",
        items=[TextInput(id="m1", text="Почта ivan@example.com")],
    )

    assert masked.placeholders_count == 1
    masked_text = masked.items[0].text
    assert "ivan@example.com" not in masked_text
    assert masked.items[0].diagnostics["elapsed_ms"] >= 0
    assert isinstance(masked.items[0].diagnostics["detector_timing_ms"], dict)

    response = await service.unmask_items(
        request_id="req-1",
        items=[TextInput(id="r1", text=f"accepted {masked_text}")],
        delete_context=True,
    )

    assert response.context_found is True
    assert "ivan@example.com" in response.items[0].text
    assert response.replacements == 1


@pytest.mark.asyncio
async def test_stream_unmask_handles_split_placeholder() -> None:
    service = _make_service()

    masked = await service.mask_items(
        request_id="req-stream",
        policy_name="external_default",
        items=[TextInput(id="m1", text="email ivan@example.com")],
    )
    placeholder_text = masked.items[0].text
    token = placeholder_text.split(" ")[-1]

    chunk1 = f"Hello {token[:5]}"
    chunk2 = token[5:] + " world"

    out1 = await service.unmask_stream_chunk(
        request_id="req-stream",
        stream_id="s1",
        chunk=chunk1,
        final=False,
    )
    out2 = await service.unmask_stream_chunk(
        request_id="req-stream",
        stream_id="s1",
        chunk=chunk2,
        final=True,
        delete_context=True,
    )

    assert out1.output == ""
    assert "ivan@example.com" in (out1.output + out2.output)


@pytest.mark.asyncio
async def test_block_policy_raises() -> None:
    service = _make_service()

    with pytest.raises(GuardrailsBlockedError):
        await service.mask_items(
            request_id="req-block",
            policy_name="strict_block",
            items=[TextInput(id="m1", text="email ivan@example.com")],
        )


@pytest.mark.asyncio
async def test_missing_context_passthrough_when_allowed() -> None:
    service = _make_service()

    response = await service.unmask_stream_chunk(
        request_id="missing",
        stream_id="s1",
        chunk="raw chunk",
        final=False,
        allow_missing_context=True,
    )
    assert response.context_found is False
    assert response.output == "raw chunk"


@pytest.mark.asyncio
async def test_passthrough_policy_returns_plain_text_without_language_resolution() -> None:
    service = _make_service()

    masked = await service.mask_items(
        request_id="req-pass",
        policy_name="onprem_passthrough",
        items=[TextInput(id="m1", text="Привет hello user@example.com")],
    )

    assert masked.mode == "passthrough"
    assert masked.findings_count == 0
    assert masked.placeholders_count == 0
    assert masked.items[0].text == "Привет hello user@example.com"
    assert masked.items[0].detections == []
