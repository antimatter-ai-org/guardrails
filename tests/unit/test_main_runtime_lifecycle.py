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


class _FakePolicyResolver:
    def __init__(self, config: Any) -> None:
        self.config = config

    def list_policies(self) -> list[str]:
        return ["external_default"]

    def resolve_policy(self, policy_name: str | None = None) -> tuple[str, Any]:
        return ("external_default", object())


@pytest.mark.asyncio
async def test_startup_schedules_background_init_in_cuda_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    redis_client = _FakeRedis()
    scheduled: list[bool] = []

    monkeypatch.setattr(main.settings, "runtime_mode", "cuda")
    monkeypatch.setattr(main.settings, "pytriton_url", "127.0.0.1:8000")
    monkeypatch.setattr(main, "apply_model_env", lambda **kwargs: None)
    monkeypatch.setattr(main.Redis, "from_url", lambda *args, **kwargs: redis_client)
    monkeypatch.setattr(main, "_load_runtime", lambda: None)
    monkeypatch.setattr(main, "_schedule_model_init", lambda: scheduled.append(True))

    await main.startup()
    assert scheduled == [True]
    assert main.app.state.embedded_pytriton_manager is None

    await main.shutdown()
    assert redis_client.closed is True


@pytest.mark.asyncio
async def test_startup_skips_embedded_pytriton_in_cpu_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    redis_client = _FakeRedis()
    scheduled: list[bool] = []

    monkeypatch.setattr(main.settings, "runtime_mode", "cpu")
    monkeypatch.setattr(main, "apply_model_env", lambda **kwargs: None)
    monkeypatch.setattr(main.Redis, "from_url", lambda *args, **kwargs: redis_client)
    monkeypatch.setattr(main, "_load_runtime", lambda: None)
    monkeypatch.setattr(main, "_schedule_model_init", lambda: scheduled.append(True))

    await main.startup()
    assert scheduled == [True]
    assert main.app.state.embedded_pytriton_manager is None
    await main.shutdown()
    assert redis_client.closed is True


@pytest.mark.asyncio
async def test_readyz_requires_embedded_pytriton_when_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.settings, "runtime_mode", "cuda")
    main.app.state.models_ready = True
    main.app.state.redis = _FakeRedis(ping_result=True)
    main.app.state.embedded_pytriton_manager = _FakeEmbeddedManager(ready=False, error="embedded not ready")

    with pytest.raises(HTTPException) as exc:
        await main.readyz()

    assert exc.value.status_code == 503
    assert "embedded not ready" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_readyz_passes_without_embedded_pytriton_in_cpu_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.settings, "runtime_mode", "cpu")
    main.app.state.models_ready = True
    main.app.state.redis = _FakeRedis(ping_result=True)
    main.app.state.embedded_pytriton_manager = None

    response: dict[str, Any] = await main.readyz()
    assert response == {"status": "ready"}


@pytest.mark.asyncio
async def test_readyz_fails_when_models_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.settings, "runtime_mode", "cpu")
    main.app.state.models_ready = False
    main.app.state.models_load_error = None
    main.app.state.redis = _FakeRedis(ping_result=True)
    main.app.state.embedded_pytriton_manager = None

    with pytest.raises(HTTPException) as exc:
        await main.readyz()

    assert exc.value.status_code == 503
    assert "model runtimes are still loading" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_background_init_sets_models_ready_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeAnalysisService:
        def ensure_profile_runtimes_ready(self, *, profile_names: list[str], timeout_s: float | None) -> dict[str, str]:
            assert timeout_s is None
            return {}

    fake_config = type("_Cfg", (), {"analyzer_profiles": {"external_rich": object()}, "recognizer_definitions": {}})()

    def fake_load_runtime() -> None:
        main.app.state.policy_resolver = _FakePolicyResolver(fake_config)
        main.app.state.analysis_service = _FakeAnalysisService()

    monkeypatch.setattr(main.settings, "runtime_mode", "cpu")
    monkeypatch.setattr(main, "_load_runtime", fake_load_runtime)

    main.app.state.models_ready = False
    await main._initialize_models_in_background()

    assert main.app.state.models_ready is True
    assert main.app.state.models_load_error is None


@pytest.mark.asyncio
async def test_background_init_records_error_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeAnalysisService:
        def ensure_profile_runtimes_ready(self, *, profile_names: list[str], timeout_s: float | None) -> dict[str, str]:
            return {"external_rich:gliner": "runtime is not ready"}

    fake_config = type("_Cfg", (), {"analyzer_profiles": {"external_rich": object()}, "recognizer_definitions": {}})()

    def fake_load_runtime() -> None:
        main.app.state.policy_resolver = _FakePolicyResolver(fake_config)
        main.app.state.analysis_service = _FakeAnalysisService()

    monkeypatch.setattr(main.settings, "runtime_mode", "cpu")
    monkeypatch.setattr(main, "_load_runtime", fake_load_runtime)

    main.app.state.models_ready = False
    await main._initialize_models_in_background()

    assert main.app.state.models_ready is False
    assert "runtime readiness check failed" in str(main.app.state.models_load_error or "")
