from __future__ import annotations

from types import SimpleNamespace

from app.config import AnalysisConfig, AnalyzerProfile, PolicyConfig, PolicyDefinition
from app.core.analysis.service import PresidioAnalysisService


class _FakeRuntime:
    def __init__(self, *, ready: bool, error: str | None = None, raises: bool = False) -> None:
        self._ready = ready
        self._error = error
        self._raises = raises
        self.calls = 0

    def ensure_ready(self, timeout_s: float) -> bool:
        self.calls += 1
        if self._raises:
            raise RuntimeError("readiness failed")
        return self._ready

    def load_error(self) -> str | None:
        return self._error


def test_ensure_profile_runtimes_ready_collects_errors() -> None:
    service = PresidioAnalysisService(
        PolicyConfig(
            default_policy="p1",
            policies={"p1": PolicyDefinition(mode="passthrough", analyzer_profile="profile-a")},
            analyzer_profiles={"profile-a": AnalyzerProfile(analysis=AnalysisConfig(recognizers=[]))},
            recognizer_definitions={},
        )
    )
    ok_runtime = _FakeRuntime(ready=True)
    bad_runtime = _FakeRuntime(ready=False, error="not ready")
    raising_runtime = _FakeRuntime(ready=False, raises=True)
    registry = SimpleNamespace(
        recognizers=[
            SimpleNamespace(name="ok", _runtime=ok_runtime),
            SimpleNamespace(name="bad", _runtime=bad_runtime),
            SimpleNamespace(name="raising", _runtime=raising_runtime),
            SimpleNamespace(name="other"),
        ]
    )

    service._registries["profile-a"] = registry  # noqa: SLF001
    errors = service.ensure_profile_runtimes_ready(profile_names=["profile-a"], timeout_s=1.0)

    assert ok_runtime.calls == 1
    assert bad_runtime.calls == 1
    assert raising_runtime.calls == 1
    assert errors["profile-a:bad"] == "not ready"
    assert "readiness failed" in errors["profile-a:raising"]
