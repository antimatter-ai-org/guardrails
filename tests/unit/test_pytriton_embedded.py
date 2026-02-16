from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from app.runtime import pytriton_embedded
from app.runtime.pytriton_embedded import EmbeddedPyTritonConfig, EmbeddedPyTritonManager


class _FakeTritonConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeTriton:
    instances: list["_FakeTriton"] = []

    def __init__(self, config):
        self.config = config
        self.bind_calls: list[dict[str, object]] = []
        self.run_calls = 0
        self.stop_calls = 0
        _FakeTriton.instances.append(self)

    def bind(self, **kwargs):
        self.bind_calls.append(kwargs)

    def run(self):
        self.run_calls += 1

    def stop(self):
        self.stop_calls += 1


def _install_fake_pytriton(monkeypatch: pytest.MonkeyPatch, triton_cls: type[_FakeTriton]) -> None:
    triton_module = types.ModuleType("pytriton.triton")
    triton_module.Triton = triton_cls
    triton_module.TritonConfig = _FakeTritonConfig
    pytriton_module = types.ModuleType("pytriton")
    pytriton_module.triton = triton_module
    monkeypatch.setitem(sys.modules, "pytriton", pytriton_module)
    monkeypatch.setitem(sys.modules, "pytriton.triton", triton_module)


def _manager_config(url: str = "127.0.0.1:8000") -> EmbeddedPyTritonConfig:
    return EmbeddedPyTritonConfig(
        pytriton_url=url,
        gliner_model_ref="gliner-model",
        token_model_ref="token-model",
        model_dir="/models",
        offline_mode=True,
        device="cuda",
        max_batch_size=16,
        enable_nemotron=True,
        grpc_port=9101,
        metrics_port=9102,
    )


def test_embedded_pytriton_manager_starts_with_loopback_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pytriton(monkeypatch, _FakeTriton)
    _FakeTriton.instances.clear()
    readiness_calls: list[tuple[str, int, float]] = []

    monkeypatch.setattr(pytriton_embedded, "apply_model_env", lambda **kwargs: None)
    monkeypatch.setattr(pytriton_embedded, "resolve_gliner_model_source", lambda **kwargs: "/models/gliner")
    monkeypatch.setattr(pytriton_embedded, "resolve_token_classifier_model_source", lambda **kwargs: "/models/token")
    monkeypatch.setattr(
        pytriton_embedded,
        "wait_for_triton_ready",
        lambda *, pytriton_url, contracts, timeout_s, poll_interval_s=0.5: readiness_calls.append(
            (pytriton_url, len(contracts), timeout_s)
        ),
    )
    monkeypatch.setattr(
        pytriton_embedded,
        "build_bindings",
        lambda **kwargs: [
            SimpleNamespace(name="gliner", infer_func=lambda **_: None, inputs=[], outputs=[], config=None),
            SimpleNamespace(name="nemotron", infer_func=lambda **_: None, inputs=[], outputs=[], config=None),
        ],
    )

    manager = EmbeddedPyTritonManager(_manager_config(url="localhost:9010"))
    manager.start()

    assert manager.client_url == "127.0.0.1:9010"
    assert manager.is_ready() is True
    assert manager.last_error() is None
    assert len(_FakeTriton.instances) == 1

    triton = _FakeTriton.instances[0]
    assert triton.run_calls == 1
    assert len(triton.bind_calls) == 2
    assert triton.config.kwargs["http_address"] == "127.0.0.1"
    assert triton.config.kwargs["grpc_address"] == "127.0.0.1"
    assert triton.config.kwargs["metrics_address"] == "127.0.0.1"
    assert triton.config.kwargs["http_port"] == 9010
    assert triton.config.kwargs["grpc_port"] == 9101
    assert triton.config.kwargs["metrics_port"] == 9102
    assert readiness_calls == [("127.0.0.1:9010", 2, 120.0)]

    manager.stop()
    manager.stop()
    assert triton.stop_calls == 1
    assert manager.is_ready() is False


def test_embedded_pytriton_manager_reports_startup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingTriton(_FakeTriton):
        def run(self):
            self.run_calls += 1
            raise RuntimeError("run failed")

    _install_fake_pytriton(monkeypatch, _FailingTriton)
    _FailingTriton.instances.clear()

    monkeypatch.setattr(pytriton_embedded, "apply_model_env", lambda **kwargs: None)
    monkeypatch.setattr(pytriton_embedded, "resolve_gliner_model_source", lambda **kwargs: "/models/gliner")
    monkeypatch.setattr(pytriton_embedded, "resolve_token_classifier_model_source", lambda **kwargs: "/models/token")
    monkeypatch.setattr(pytriton_embedded, "wait_for_triton_ready", lambda **kwargs: None)
    monkeypatch.setattr(
        pytriton_embedded,
        "build_bindings",
        lambda **kwargs: [SimpleNamespace(name="gliner", infer_func=lambda **_: None, inputs=[], outputs=[], config=None)],
    )

    manager = EmbeddedPyTritonManager(_manager_config())
    with pytest.raises(RuntimeError, match="PyTriton startup failed"):
        manager.start()

    assert manager.is_ready() is False
    assert manager.last_error() is not None
    assert len(_FailingTriton.instances) == 1
    assert _FailingTriton.instances[0].stop_calls == 1


def test_embedded_pytriton_manager_rejects_non_loopback_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pytriton(monkeypatch, _FakeTriton)
    manager = EmbeddedPyTritonManager(_manager_config(url="10.10.10.10:8000"))

    with pytest.raises(ValueError, match="requires loopback"):
        manager.start()

    assert manager.is_ready() is False
    assert manager.last_error() is not None
