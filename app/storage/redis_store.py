from __future__ import annotations

from typing import Any

import orjson
from redis.asyncio import Redis


class RedisMappingStore:
    def __init__(
        self,
        redis_client: Redis,
        mapping_prefix: str = "gr:map",
        stream_prefix: str = "gr:stream",
    ) -> None:
        self._redis = redis_client
        self._mapping_prefix = mapping_prefix
        self._stream_prefix = stream_prefix

    def _mapping_key(self, request_id: str) -> str:
        return f"{self._mapping_prefix}:{request_id}"

    def _stream_key(self, request_id: str, stream_id: str) -> str:
        return f"{self._stream_prefix}:{request_id}:{stream_id}"

    async def save_mapping(self, request_id: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        await self._redis.set(self._mapping_key(request_id), orjson.dumps(payload), ex=ttl_seconds)

    async def load_mapping(self, request_id: str) -> dict[str, Any] | None:
        raw = await self._redis.get(self._mapping_key(request_id))
        if raw is None:
            return None
        return orjson.loads(raw)

    async def delete_mapping(self, request_id: str) -> None:
        await self._redis.delete(self._mapping_key(request_id))

    async def save_stream_state(
        self,
        request_id: str,
        stream_id: str,
        payload: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        await self._redis.set(self._stream_key(request_id, stream_id), orjson.dumps(payload), ex=ttl_seconds)

    async def load_stream_state(self, request_id: str, stream_id: str) -> dict[str, Any] | None:
        raw = await self._redis.get(self._stream_key(request_id, stream_id))
        if raw is None:
            return None
        return orjson.loads(raw)

    async def delete_stream_state(self, request_id: str, stream_id: str) -> None:
        await self._redis.delete(self._stream_key(request_id, stream_id))

    async def delete_all_stream_states(self, request_id: str) -> int:
        pattern = self._stream_key(request_id, "*")
        keys = [key async for key in self._redis.scan_iter(match=pattern, count=1000)]
        if not keys:
            return 0
        return int(await self._redis.delete(*keys))

    async def delete_all_for_request(self, request_id: str) -> None:
        await self.delete_mapping(request_id)
        await self.delete_all_stream_states(request_id)
