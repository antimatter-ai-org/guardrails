from __future__ import annotations

from typing import Any

import orjson
from redis.asyncio import Redis


class RedisMappingStore:
    def __init__(self, redis_client: Redis, key_prefix: str = "gr:map") -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix

    def _key(self, request_id: str) -> str:
        return f"{self._key_prefix}:{request_id}"

    async def save(self, request_id: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        await self._redis.set(self._key(request_id), orjson.dumps(payload), ex=ttl_seconds)

    async def load(self, request_id: str) -> dict[str, Any] | None:
        raw = await self._redis.get(self._key(request_id))
        if raw is None:
            return None
        return orjson.loads(raw)

    async def delete(self, request_id: str) -> None:
        await self._redis.delete(self._key(request_id))
