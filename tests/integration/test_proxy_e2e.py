from __future__ import annotations

import os
import time

import httpx
import pytest

GUARDRAILS_URL = os.getenv("GUARDRAILS_BASE_URL", "http://localhost:8080")
MOCK_URL = os.getenv("MOCK_LLM_BASE_URL", "http://localhost:8090")


def _wait_ready(url: str, timeout_seconds: float = 20) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = httpx.get(f"{url}/health", timeout=1.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"Service not ready: {url}")


@pytest.fixture(scope="session", autouse=True)
def _ready() -> None:
    _wait_ready(GUARDRAILS_URL)
    _wait_ready(MOCK_URL)


def test_external_model_masks_before_upstream_and_unmasks_after() -> None:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "Мой email ivan.petrov@example.com, паспорт 1234 567890",
            }
        ],
    }

    response = httpx.post(
        f"{GUARDRAILS_URL}/v1/chat/completions",
        json=payload,
        headers={"x-upstream-base-url": MOCK_URL, "x-request-id": "int-req-1"},
        timeout=10.0,
    )
    assert response.status_code == 200, response.text

    body = response.json()
    text = body["choices"][0]["message"]["content"]
    assert "ivan.petrov@example.com" in text
    assert "1234 567890" in text

    masked_values = int(response.headers.get("x-guardrails-masked-values", "0"))
    assert masked_values >= 1

    upstream_payload = httpx.get(f"{MOCK_URL}/debug/last-request", timeout=5.0).json()["payload"]
    upstream_text = upstream_payload["messages"][0]["content"]
    assert "ivan.petrov@example.com" not in upstream_text
    assert "1234 567890" not in upstream_text
    assert "<GR" in upstream_text


def test_onprem_model_passthrough() -> None:
    payload = {
        "model": "onprem/llama-3.1-70b",
        "messages": [
            {
                "role": "user",
                "content": "email admin@example.com",
            }
        ],
    }

    response = httpx.post(
        f"{GUARDRAILS_URL}/v1/chat/completions",
        json=payload,
        headers={"x-upstream-base-url": MOCK_URL, "x-request-id": "int-req-2"},
        timeout=10.0,
    )
    assert response.status_code == 200, response.text
    assert response.headers.get("x-guardrails-masked-values") == "0"

    upstream_payload = httpx.get(f"{MOCK_URL}/debug/last-request", timeout=5.0).json()["payload"]
    upstream_text = upstream_payload["messages"][0]["content"]
    assert "admin@example.com" in upstream_text
