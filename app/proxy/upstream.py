from __future__ import annotations

from typing import Any

import httpx


class UpstreamClient:
    def __init__(self, timeout_seconds: float = 60.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def chat_completions(
        self,
        base_url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        response = await self._client.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        content_type = response.headers.get("content-type", "")
        body: dict[str, Any]
        if "application/json" in content_type:
            body = response.json()
        else:
            body = {
                "error": {
                    "message": response.text,
                    "type": "upstream_error",
                    "status_code": response.status_code,
                }
            }

        passthrough_headers = {
            "x-request-id": response.headers.get("x-request-id", ""),
        }
        return response.status_code, body, passthrough_headers
