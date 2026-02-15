from __future__ import annotations

import types

import pytest

from app.eval import run as eval_run


class _FakeRuntime:
    def __init__(self, *, ready: bool, error: str | None = None):
        self._ready = ready
        self._error = error
        self.calls: list[float] = []

    def warm_up(self, timeout_s: float) -> bool:
        self.calls.append(timeout_s)
        return self._ready

    def is_ready(self) -> bool:
        return self._ready

    def load_error(self) -> str | None:
        return self._error


class _FakeRecognizer:
    def __init__(self, name: str, runtime: _FakeRuntime | None):
        self.name = name
        self._runtime = runtime

    def get_supported_entities(self) -> list[str]:
        return ["PERSON"]

    def analyze(self, text: str, entities: list[str], nlp_artifacts=None):
        return []


class _FakeService:
    def __init__(self, recognizers: list[_FakeRecognizer]):
        self._config = types.SimpleNamespace(analyzer_profiles={"profile": object()})
        self._recognizers = recognizers

    def _requires_analyzer_engine(self, profile) -> bool:  # noqa: ARG002
        return False

    def _get_registry(self, profile_name: str):  # noqa: ARG002
        return types.SimpleNamespace(recognizers=self._recognizers)


def test_warm_up_profile_collects_runtime_statuses() -> None:
    runtime_ok = _FakeRuntime(ready=True)
    runtime_bad = _FakeRuntime(ready=False, error="init failed")
    recognizers = [
        _FakeRecognizer("gliner:ru", runtime_ok),
        _FakeRecognizer("gliner:en", runtime_bad),
    ]
    service = _FakeService(recognizers)

    statuses = eval_run._warm_up_profile(service, "profile", timeout_seconds=3.0)

    assert len(statuses) == 2
    assert statuses[0]["recognizer"] == "gliner:ru"
    assert statuses[0]["ready"] is True
    assert statuses[1]["recognizer"] == "gliner:en"
    assert statuses[1]["ready"] is False
    assert statuses[1]["load_error"] == "init failed"
    assert runtime_ok.calls == [3.0]
    assert runtime_bad.calls == [3.0]


def test_handle_warmup_failures_returns_failures_in_non_strict_mode() -> None:
    failures = eval_run._handle_warmup_failures(
        [
            {"recognizer": "a", "has_runtime": True, "ready": True},
            {"recognizer": "b", "has_runtime": True, "ready": False},
        ],
        strict=False,
    )
    assert len(failures) == 1
    assert failures[0]["recognizer"] == "b"


def test_handle_warmup_failures_raises_in_strict_mode() -> None:
    with pytest.raises(RuntimeError):
        eval_run._handle_warmup_failures(
            [{"recognizer": "gliner:ru", "has_runtime": True, "ready": False}],
            strict=True,
        )
