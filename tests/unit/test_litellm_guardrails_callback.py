from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest
from fastapi import HTTPException


def _load_callback_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "integrations"
        / "litellm_openrouter"
        / "custom_callbacks"
        / "guardrails_callback.py"
    )
    spec = importlib.util.spec_from_file_location("litellm_guardrails_callback", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load guardrails callback module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_pre_call_masks_input_and_stores_session_ctx() -> None:
    module = _load_callback_module()
    callback = module.GuardrailsCallback()

    async def fake_apply(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["source"] == "INPUT"
        assert kwargs["mode"] == "DEIDENTIFY"
        return {
            "outputs": [
                {"id": "msg-0", "text": "hello <MASK_A>"},
                {"id": "msg-1-part-0", "text": "copy <MASK_B>"},
            ],
            "session": {"id": "sess_abc"},
            "policy_id": "external",
        }

    callback._guardrails_apply = fake_apply  # type: ignore[method-assign]

    payload = {
        "messages": [
            {"role": "user", "content": "hello user@example.com"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "copy 4510 667788"},
                    {"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}},
                ],
            },
        ]
    }

    transformed = await callback.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=payload,
        call_type="acompletion",
    )

    assert transformed["messages"][0]["content"] == "hello <MASK_A>"
    assert transformed["messages"][1]["content"][0]["text"] == "copy <MASK_B>"
    metadata = transformed["metadata"]
    assert metadata["_guardrails_ctx"]["session_id"] == "sess_abc"


@pytest.mark.asyncio
async def test_post_call_non_stream_unmasks_and_finalizes() -> None:
    module = _load_callback_module()
    callback = module.GuardrailsCallback()
    finalized: list[str] = []

    async def fake_apply(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["source"] == "OUTPUT"
        assert kwargs["mode"] == "REIDENTIFY"
        assert kwargs["session_id"] == "sess_non_stream"
        return {
            "outputs": [{"id": "choice-0", "text": "final ivan.petrov@example.com"}],
        }

    async def fake_finalize(session_id: str) -> None:
        finalized.append(session_id)

    callback._guardrails_apply = fake_apply  # type: ignore[method-assign]
    callback._finalize_best_effort = fake_finalize  # type: ignore[method-assign]

    request_data = {"metadata": {"_guardrails_ctx": {"session_id": "sess_non_stream", "request_id": "rid-1"}}}
    response = {"choices": [{"message": {"content": "final <MASK_EMAIL>"}}]}

    mutated = await callback.async_post_call_success_hook(
        data=request_data,
        user_api_key_dict=None,
        response=response,
    )

    assert mutated["choices"][0]["message"]["content"] == "final ivan.petrov@example.com"
    assert "_guardrails_ctx" not in request_data["metadata"]
    assert finalized == ["sess_non_stream"]


@pytest.mark.asyncio
async def test_streaming_hook_unmasks_chunks_and_flushes_tail() -> None:
    module = _load_callback_module()
    callback = module.GuardrailsCallback()
    finalized: list[str] = []
    calls: list[tuple[str, bool, str]] = []

    async def fake_stream_apply(**kwargs: Any) -> dict[str, Any]:
        calls.append((kwargs["stream_id"], bool(kwargs["final"]), kwargs["chunk"]))
        key = (kwargs["stream_id"], bool(kwargs["final"]), kwargs["chunk"])
        mapping = {
            ("choice-0", False, "<MA"): {"output_chunk": ""},
            ("choice-0", False, "SK>"): {"output_chunk": "secret"},
            ("choice-0", True, ""): {"output_chunk": "!"},
        }
        return mapping[key]

    async def fake_finalize(session_id: str) -> None:
        finalized.append(session_id)

    callback._guardrails_apply_stream = fake_stream_apply  # type: ignore[method-assign]
    callback._finalize_best_effort = fake_finalize  # type: ignore[method-assign]

    async def upstream():
        yield {
            "id": "chunk-1",
            "model": "demo-openrouter",
            "choices": [{"index": 0, "delta": {"content": "<MA"}, "finish_reason": None}],
        }
        yield {
            "id": "chunk-2",
            "model": "demo-openrouter",
            "choices": [{"index": 0, "delta": {"content": "SK>"}, "finish_reason": "stop"}],
        }

    request_data = {"metadata": {"_guardrails_ctx": {"session_id": "sess_stream", "request_id": "rid-stream"}}}
    out: list[dict[str, Any]] = []
    async for chunk in callback.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None,
        response=upstream(),
        request_data=request_data,
    ):
        out.append(chunk)

    assert out[0]["choices"][0]["delta"]["content"] == ""
    assert out[1]["choices"][0]["delta"]["content"] == "secret"
    assert out[2]["choices"][0]["delta"]["content"] == "!"
    assert calls == [
        ("choice-0", False, "<MA"),
        ("choice-0", False, "SK>"),
        ("choice-0", True, ""),
    ]
    assert finalized == ["sess_stream"]
    assert "_guardrails_ctx" not in request_data["metadata"]


@pytest.mark.asyncio
async def test_fail_closed_when_guardrails_apply_fails() -> None:
    module = _load_callback_module()
    callback = module.GuardrailsCallback()

    async def failing_apply(**kwargs: Any) -> dict[str, Any]:
        raise HTTPException(status_code=503, detail="guardrails unavailable")

    callback._guardrails_apply = failing_apply  # type: ignore[method-assign]

    with pytest.raises(HTTPException) as exc:
        await callback.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data={"messages": [{"role": "user", "content": "email user@example.com"}]},
            call_type="acompletion",
        )

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_missing_or_invalid_session_paths() -> None:
    module = _load_callback_module()
    callback = module.GuardrailsCallback()

    response = {"choices": [{"message": {"content": "hello"}}]}
    passthrough = await callback.async_post_call_success_hook(
        data={"metadata": {}},
        user_api_key_dict=None,
        response=response,
    )
    assert passthrough is response

    with pytest.raises(HTTPException) as exc:
        await callback.async_post_call_success_hook(
            data={"metadata": {"_guardrails_ctx": {"session_id": None}}},
            user_api_key_dict=None,
            response=response,
        )

    assert exc.value.status_code == 503
