from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from app import main


class _FakeRedis:
    def __init__(self, *, ping_result: bool = True) -> None:
        self._ping_result = ping_result
        self.closed = False

    async def ping(self) -> bool:
        return self._ping_result

    async def aclose(self) -> None:
        self.closed = True


class _FakeEmbeddedManager:
    def __init__(self, *, ready: bool = True, error: str | None = None, client_url: str = "127.0.0.1:8000") -> None:
        self.ready = ready
        self.error = error
        self.client_url = client_url
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def is_ready(self) -> bool:
        return self.ready

    def last_error(self) -> str | None:
        return self.error


@pytest.mark.asyncio
async def test_startup_starts_embedded_pytriton_in_cuda_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    redis_client = _FakeRedis()
    manager = _FakeEmbeddedManager(client_url="127.0.0.1:9011")

    monkeypatch.setattr(main.settings, "runtime_mode", "cuda")
    monkeypatch.setattr(main.settings, "pytriton_url", "127.0.0.1:8000")
    monkeypatch.setattr(main, "apply_model_env", lambda **kwargs: None)
    monkeypatch.setattr(main.Redis, "from_url", lambda *args, **kwargs: redis_client)
    monkeypatch.setattr(main, "_build_embedded_pytriton_manager", lambda: manager)
    monkeypatch.setattr(main, "_load_runtime", lambda: None)

    await main.startup()
    assert manager.start_calls == 1
    assert main.app.state.embedded_pytriton_manager is manager
    assert main.settings.pytriton_url == "127.0.0.1:9011"

    await main.shutdown()
    assert manager.stop_calls == 1
    assert redis_client.closed is True


@pytest.mark.asyncio
async def test_startup_skips_embedded_pytriton_in_cpu_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    redis_client = _FakeRedis()

    monkeypatch.setattr(main.settings, "runtime_mode", "cpu")
    monkeypatch.setattr(main, "apply_model_env", lambda **kwargs: None)
    monkeypatch.setattr(main.Redis, "from_url", lambda *args, **kwargs: redis_client)
    monkeypatch.setattr(main, "_build_embedded_pytriton_manager", lambda: (_ for _ in ()).throw(AssertionError("should not build manager")))
    monkeypatch.setattr(main, "_load_runtime", lambda: None)

    await main.startup()
    assert main.app.state.embedded_pytriton_manager is None
    await main.shutdown()
    assert redis_client.closed is True


@pytest.mark.asyncio
async def test_readyz_requires_embedded_pytriton_when_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.settings, "runtime_mode", "cuda")
    main.app.state.redis = _FakeRedis(ping_result=True)
    main.app.state.embedded_pytriton_manager = _FakeEmbeddedManager(ready=False, error="embedded not ready")

    with pytest.raises(HTTPException) as exc:
        await main.readyz()

    assert exc.value.status_code == 503
    assert "embedded not ready" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_readyz_passes_without_embedded_pytriton_in_cpu_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.settings, "runtime_mode", "cpu")
    main.app.state.redis = _FakeRedis(ping_result=True)
    main.app.state.embedded_pytriton_manager = None

    response: dict[str, Any] = await main.readyz()
    assert response == {"status": "ready"}
