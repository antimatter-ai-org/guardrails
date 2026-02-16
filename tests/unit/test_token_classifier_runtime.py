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


def test_local_runtime_ensure_ready_reflects_constructor_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_load(self) -> None:  # noqa: ANN001
        self._pipeline = None  # noqa: SLF001
        self._load_error = "load failed"  # noqa: SLF001

    monkeypatch.setattr(token_classifier_runtime.LocalCpuTokenClassifierRuntime, "_load_model", failed_load)
    runtime = token_classifier_runtime.LocalCpuTokenClassifierRuntime(model_name="dummy")

    assert runtime.ensure_ready(timeout_s=0.5) is False
    assert runtime.is_ready() is False
    assert runtime.load_error() == "load failed"


def test_pytriton_runtime_predict_requires_readiness_check() -> None:
    runtime = token_classifier_runtime.PyTritonTokenClassifierRuntime(
        model_name="nemotron",
        pytriton_url="localhost:8000",
        init_timeout_s=10.0,
        infer_timeout_s=20.0,
    )

    with pytest.raises(RuntimeError, match="not ready"):
        runtime.predict_entities("hello", ["email"], threshold=0.5)


def test_pytriton_runtime_ensure_ready_success_sets_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[tuple[str, int]] = []

    def fake_wait(*, pytriton_url: str, contracts: list[object], timeout_s: float, poll_interval_s: float = 0.5) -> None:
        called.append((pytriton_url, len(contracts)))

    monkeypatch.setattr(token_classifier_runtime, "wait_for_triton_ready", fake_wait)
    runtime = token_classifier_runtime.PyTritonTokenClassifierRuntime(
        model_name="nemotron",
        pytriton_url="localhost:8000",
        init_timeout_s=10.0,
        infer_timeout_s=20.0,
    )

    assert runtime.ensure_ready(timeout_s=1.0) is True
    assert runtime.is_ready() is True
    assert runtime.load_error() is None
    assert called == [("localhost:8000", 1)]
