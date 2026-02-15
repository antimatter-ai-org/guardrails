from __future__ import annotations

import pytest

from app.runtime import token_classifier_runtime


class _SentinelRuntime:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_token_classifier_runtime_cpu_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(token_classifier_runtime, "LocalCpuTokenClassifierRuntime", _SentinelRuntime)

    runtime = token_classifier_runtime.build_token_classifier_runtime(
        runtime_mode="cpu",
        model_name="scanpatch/pii-ner-nemotron",
        cpu_device="auto",
        pytriton_url="localhost:8000",
        pytriton_model_name="nemotron",
        pytriton_init_timeout_s=20.0,
        pytriton_infer_timeout_s=30.0,
        aggregation_strategy="simple",
    )

    assert isinstance(runtime, _SentinelRuntime)
    assert runtime.kwargs["model_name"] == "scanpatch/pii-ner-nemotron"
    assert runtime.kwargs["preferred_device"] == "auto"


def test_build_token_classifier_runtime_cuda_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(token_classifier_runtime, "PyTritonTokenClassifierRuntime", _SentinelRuntime)

    runtime = token_classifier_runtime.build_token_classifier_runtime(
        runtime_mode="cuda",
        model_name="ignored-local-name",
        cpu_device="auto",
        pytriton_url="localhost:8000",
        pytriton_model_name="nemotron",
        pytriton_init_timeout_s=20.0,
        pytriton_infer_timeout_s=30.0,
        aggregation_strategy="simple",
    )

    assert isinstance(runtime, _SentinelRuntime)
    assert runtime.kwargs["model_name"] == "nemotron"
    assert runtime.kwargs["pytriton_url"] == "localhost:8000"


def test_build_token_classifier_runtime_invalid_mode() -> None:
    with pytest.raises(ValueError):
        token_classifier_runtime.build_token_classifier_runtime(
            runtime_mode="invalid",
            model_name="x",
            cpu_device="auto",
            pytriton_url="localhost:8000",
            pytriton_model_name="nemotron",
            pytriton_init_timeout_s=20.0,
            pytriton_infer_timeout_s=30.0,
        )


def test_normalize_token_classifier_label() -> None:
    assert token_classifier_runtime._normalize_token_classifier_label("B-email") == "email"  # noqa: SLF001
    assert token_classifier_runtime._normalize_token_classifier_label("I-address_city") == "address_city"  # noqa: SLF001
    assert token_classifier_runtime._normalize_token_classifier_label("organization") == "organization"  # noqa: SLF001
