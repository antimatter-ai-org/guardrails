from __future__ import annotations

import pytest

from app.runtime import torch_runtime


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeMps:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = _FakeMps(mps_available)


class _FakeTorch:
    def __init__(self, cuda_available: bool, mps_available: bool) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self.backends = _FakeBackends(mps_available)


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_runtime, "_load_torch", lambda: _FakeTorch(cuda_available=True, mps_available=True))
    assert torch_runtime.resolve_torch_device("auto") == "cuda"

    monkeypatch.setattr(torch_runtime, "_load_torch", lambda: _FakeTorch(cuda_available=False, mps_available=True))
    assert torch_runtime.resolve_torch_device("auto") == "mps"

    monkeypatch.setattr(torch_runtime, "_load_torch", lambda: _FakeTorch(cuda_available=False, mps_available=False))
    assert torch_runtime.resolve_torch_device("auto") == "cpu"


def test_cpu_runtime_auto_prefers_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_runtime, "_load_torch", lambda: _FakeTorch(cuda_available=True, mps_available=True))
    assert torch_runtime.resolve_cpu_runtime_device("auto") == "mps"

    monkeypatch.setattr(torch_runtime, "_load_torch", lambda: _FakeTorch(cuda_available=True, mps_available=False))
    assert torch_runtime.resolve_cpu_runtime_device("auto") == "cpu"

    monkeypatch.setattr(torch_runtime, "_load_torch", lambda: _FakeTorch(cuda_available=False, mps_available=False))
    assert torch_runtime.resolve_cpu_runtime_device("auto") == "cpu"


def test_auto_without_torch_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_runtime, "_load_torch", lambda: None)
    assert torch_runtime.resolve_torch_device("auto") == "cpu"
    assert torch_runtime.resolve_cpu_runtime_device("auto") == "cpu"


def test_explicit_unavailable_device_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_runtime, "_load_torch", lambda: _FakeTorch(cuda_available=False, mps_available=False))

    with pytest.raises(RuntimeError):
        torch_runtime.resolve_torch_device("cuda")

    with pytest.raises(RuntimeError):
        torch_runtime.resolve_torch_device("mps")


def test_invalid_device_raises() -> None:
    with pytest.raises(ValueError):
        torch_runtime.resolve_torch_device("bad-device")
    with pytest.raises(ValueError):
        torch_runtime.resolve_cpu_runtime_device("bad-device")
