from __future__ import annotations

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


def test_build_gliner_runtime_gpu_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gliner_runtime, "PyTritonGlinerRuntime", _SentinelRuntime)

    runtime = gliner_runtime.build_gliner_runtime(
        runtime_mode="gpu",
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
