from __future__ import annotations

import sys
import threading
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


def test_local_cpu_runtime_warm_up_times_out_when_model_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = gliner_runtime.LocalCpuGlinerRuntime(model_name="dummy")
    gate = threading.Event()

    def delayed_load() -> None:
        gate.wait(timeout=0.5)
        runtime._model = object()  # noqa: SLF001

    monkeypatch.setattr(runtime, "_load_model", delayed_load)

    assert runtime.warm_up(timeout_s=0.0) is False
    assert runtime.is_ready() is False


def test_local_cpu_runtime_warm_up_sets_ready_state(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = gliner_runtime.LocalCpuGlinerRuntime(model_name="dummy")

    def instant_load() -> None:
        runtime._model = object()  # noqa: SLF001

    monkeypatch.setattr(runtime, "_load_model", instant_load)

    assert runtime.warm_up(timeout_s=0.5) is True
    assert runtime.is_ready() is True
    assert runtime.load_error() is None


def test_pytriton_runtime_warm_up_success_sets_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModelClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def infer_batch(self, *, text, labels_json, threshold):
            payload = np.array([[b"[]"]], dtype=object)
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

    assert runtime.warm_up(timeout_s=1.0) is True
    assert runtime.is_ready() is True
    assert runtime.load_error() is None
