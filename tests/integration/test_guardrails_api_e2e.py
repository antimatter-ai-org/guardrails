from __future__ import annotations

import os
import re
import time

import httpx
import pytest

GUARDRAILS_URL = os.getenv("GUARDRAILS_BASE_URL", "http://localhost:8080")


def _wait_ready(url: str, timeout_seconds: float = 20) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = httpx.get(f"{url}/healthz", timeout=1.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"Service not ready: {url}")


def test_apply_deidentify_and_stream_reidentify_flow() -> None:
    try:
        _wait_ready(GUARDRAILS_URL)
    except RuntimeError:
        pytest.skip("guardrails service is not running")

    apply_payload = {
        "policy_id": "external_default",
        "source": "INPUT",
        "content": [
            {
                "id": "msg-1",
                "text": "Мой email ivan.petrov@example.com, паспорт 1234 567890",
            }
        ],
        "transforms": [{"type": "reversible_mask", "mode": "DEIDENTIFY"}],
    }

    masked_response = httpx.post(
        f"{GUARDRAILS_URL}/v1/guardrails/apply",
        json=apply_payload,
        timeout=10.0,
    )
    assert masked_response.status_code == 200, masked_response.text

    masked_body = masked_response.json()
    masked_text = masked_body["outputs"][0]["text"]
    assert "ivan.petrov@example.com" not in masked_text
    assert "1234 567890" not in masked_text
    session_id = masked_body["session"]["id"]

    match = re.search(r"<[^>]+>", masked_text)
    assert match is not None
    first_placeholder = match.group(0)
    simulated = f"Ответ: {first_placeholder} подтвержден"

    split_idx = max(1, len(first_placeholder) // 2)
    chunk1 = simulated[: 7 + split_idx]
    chunk2 = simulated[7 + split_idx :]

    stream_1 = httpx.post(
        f"{GUARDRAILS_URL}/v1/guardrails/apply-stream",
        json={
            "policy_id": "external_default",
            "source": "OUTPUT",
            "transforms": [
                {
                    "type": "reversible_mask",
                    "mode": "REIDENTIFY",
                    "session": {"id": session_id},
                }
            ],
            "stream": {
                "id": "choice-0",
                "chunk": chunk1,
                "final": False,
            },
        },
        timeout=10.0,
    )
    assert stream_1.status_code == 200, stream_1.text

    stream_2 = httpx.post(
        f"{GUARDRAILS_URL}/v1/guardrails/apply-stream",
        json={
            "policy_id": "external_default",
            "source": "OUTPUT",
            "transforms": [
                {
                    "type": "reversible_mask",
                    "mode": "REIDENTIFY",
                    "session": {"id": session_id},
                }
            ],
            "stream": {
                "id": "choice-0",
                "chunk": chunk2,
                "final": True,
            },
        },
        timeout=10.0,
    )
    assert stream_2.status_code == 200, stream_2.text

    output = stream_1.json()["output_chunk"] + stream_2.json()["output_chunk"]
    assert "ivan.petrov@example.com" in output

    finalize = httpx.post(
        f"{GUARDRAILS_URL}/v1/guardrails/sessions/{session_id}/finalize",
        timeout=10.0,
    )
    assert finalize.status_code == 200, finalize.text
    assert finalize.json()["context_deleted"] is True


def test_block_policy_returns_blocked_action() -> None:
    try:
        _wait_ready(GUARDRAILS_URL)
    except RuntimeError:
        pytest.skip("guardrails service is not running")

    response = httpx.post(
        f"{GUARDRAILS_URL}/v1/guardrails/apply",
        json={
            "policy_id": "strict_block",
            "source": "INPUT",
            "content": [{"id": "msg-1", "text": "user email admin@example.com"}],
            "transforms": [{"type": "reversible_mask", "mode": "DEIDENTIFY"}],
        },
        timeout=10.0,
    )
    assert response.status_code == 200
    assert response.json()["action"] == "BLOCKED"
