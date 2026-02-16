from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from app.runtime import gliner_runtime


class _SentinelRuntime:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_build_gliner_runtime_cpu_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gliner_runtime, "LocalCpuGlinerRuntime", _SentinelRuntime)

    runtime = gliner_runtime.build_gliner_runtime(
        runtime_mode="cpu",
        model_name="model-a",
        cpu_device="auto",
        pytriton_url="pytriton",
        pytriton_model_name="gliner",
        pytriton_init_timeout_s=10.0,
        pytriton_infer_timeout_s=20.0,
    )

    assert isinstance(runtime, _SentinelRuntime)
    assert runtime.kwargs["model_name"] == "model-a"


def test_build_gliner_runtime_cuda_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gliner_runtime, "PyTritonGlinerRuntime", _SentinelRuntime)

    runtime = gliner_runtime.build_gliner_runtime(
        runtime_mode="cuda",
        model_name="unused-hf-name",
        cpu_device="auto",
        pytriton_url="pytriton",
        pytriton_model_name="gliner",
        pytriton_init_timeout_s=10.0,
        pytriton_infer_timeout_s=20.0,
    )

    assert isinstance(runtime, _SentinelRuntime)
    assert runtime.kwargs["model_name"] == "gliner"
    assert runtime.kwargs["pytriton_url"] == "pytriton"


def test_build_gliner_runtime_invalid_mode() -> None:
    with pytest.raises(ValueError):
        gliner_runtime.build_gliner_runtime(
            runtime_mode="invalid",
            model_name="a",
            cpu_device="auto",
            pytriton_url="pytriton",
            pytriton_model_name="gliner",
            pytriton_init_timeout_s=10.0,
            pytriton_infer_timeout_s=20.0,
        )


def test_pytriton_runtime_chunks_requests_to_max_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class FakeModelClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def infer_batch(self, *, text, labels_json, threshold):
            calls.append(int(text.shape[0]))
            payload = np.array([[b"[]"] for _ in range(int(text.shape[0]))], dtype=object)
            return {"detections_json": payload}

    fake_client_module = types.ModuleType("pytriton.client")
    fake_client_module.ModelClient = FakeModelClient
    fake_package = types.ModuleType("pytriton")
    fake_package.client = fake_client_module

    monkeypatch.setitem(sys.modules, "pytriton", fake_package)
    monkeypatch.setitem(sys.modules, "pytriton.client", fake_client_module)

    runtime = gliner_runtime.PyTritonGlinerRuntime(
        model_name="gliner",
        pytriton_url="localhost:8000",
        init_timeout_s=10.0,
        infer_timeout_s=20.0,
    )
    outputs = runtime._predict_batch(texts=["x"] * 70, labels=["person"], threshold=0.5)

    assert calls == [32, 32, 6]
    assert len(outputs) == 70


def test_pytriton_runtime_adapts_when_server_requires_smaller_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class FakeModelClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def infer_batch(self, *, text, labels_json, threshold):
            batch_size = int(text.shape[0])
            calls.append(batch_size)
            if batch_size > 16:
                raise RuntimeError("[request id: 0] inference request batch-size must be <= 16 for 'gliner'")
            payload = np.array([[b"[]"] for _ in range(batch_size)], dtype=object)
            return {"detections_json": payload}

    fake_client_module = types.ModuleType("pytriton.client")
    fake_client_module.ModelClient = FakeModelClient
    fake_package = types.ModuleType("pytriton")
    fake_package.client = fake_client_module

    monkeypatch.setitem(sys.modules, "pytriton", fake_package)
    monkeypatch.setitem(sys.modules, "pytriton.client", fake_client_module)

    runtime = gliner_runtime.PyTritonGlinerRuntime(
        model_name="gliner",
        pytriton_url="localhost:8000",
        init_timeout_s=10.0,
        infer_timeout_s=20.0,
    )
    outputs = runtime._predict_batch(texts=["x"] * 40, labels=["person"], threshold=0.5)

    assert calls[0] == 32
    assert calls[1:] == [16, 16, 8]
    assert len(outputs) == 40


def test_local_cpu_runtime_ensure_ready_reflects_constructor_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_load(self) -> None:  # noqa: ANN001
        self._model = None  # noqa: SLF001
        self._load_error = "load failed"  # noqa: SLF001

    monkeypatch.setattr(gliner_runtime.LocalCpuGlinerRuntime, "_load_model", failed_load)
    runtime = gliner_runtime.LocalCpuGlinerRuntime(model_name="dummy")

    assert runtime.ensure_ready(timeout_s=0.5) is False
    assert runtime.is_ready() is False
    assert runtime.load_error() == "load failed"


def test_local_cpu_runtime_ensure_ready_is_ready_after_constructor_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def successful_load(self) -> None:  # noqa: ANN001
        self._model = object()  # noqa: SLF001
        self._load_error = None  # noqa: SLF001

    monkeypatch.setattr(gliner_runtime.LocalCpuGlinerRuntime, "_load_model", successful_load)
    runtime = gliner_runtime.LocalCpuGlinerRuntime(model_name="dummy")

    assert runtime.ensure_ready(timeout_s=0.5) is True
    assert runtime.is_ready() is True
    assert runtime.load_error() is None


def test_pytriton_runtime_predict_requires_readiness_check() -> None:
    runtime = gliner_runtime.PyTritonGlinerRuntime(
        model_name="gliner",
        pytriton_url="localhost:8000",
        init_timeout_s=10.0,
        infer_timeout_s=20.0,
    )

    with pytest.raises(RuntimeError, match="not ready"):
        runtime.predict_entities("hello", ["person"], threshold=0.5)


def test_pytriton_runtime_ensure_ready_success_sets_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[tuple[str, int]] = []

    def fake_wait(*, pytriton_url: str, contracts: list[object], timeout_s: float, poll_interval_s: float = 0.5) -> None:
        called.append((pytriton_url, len(contracts)))

    monkeypatch.setattr(gliner_runtime, "wait_for_triton_ready", fake_wait)
    runtime = gliner_runtime.PyTritonGlinerRuntime(
        model_name="gliner",
        pytriton_url="localhost:8000",
        init_timeout_s=10.0,
        infer_timeout_s=20.0,
    )

    assert runtime.ensure_ready(timeout_s=1.0) is True
    assert runtime.is_ready() is True
    assert runtime.load_error() is None
    assert called == [("localhost:8000", 1)]
