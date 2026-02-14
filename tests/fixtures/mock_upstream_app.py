from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI

app = FastAPI(title="mock-upstream-llm")

_last_request: dict[str, Any] | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug/last-request")
async def last_request() -> dict[str, Any]:
    return {"payload": _last_request}


@app.post("/v1/chat/completions")
async def chat(payload: dict[str, Any]) -> dict[str, Any]:
    global _last_request
    _last_request = payload

    content = ""
    messages = payload.get("messages", [])
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict):
            raw = last.get("content")
            if isinstance(raw, str):
                content = raw
            elif isinstance(raw, list):
                parts: list[str] = []
                for item in raw:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                content = " ".join(parts)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", "mock-model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"Processed: {content}"},
                "finish_reason": "stop",
            }
        ],
    }
