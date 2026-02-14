from __future__ import annotations

from typing import Any

_SUPPORTED_DEVICES = {"auto", "cpu", "cuda", "mps"}
_SUPPORTED_CPU_RUNTIME_DEVICES = {"auto", "cpu", "mps"}


def _load_torch() -> Any | None:
    try:
        import torch
    except Exception:
        return None
    return torch


def _is_mps_available(torch_module: Any) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    return False


def resolve_torch_device(preferred_device: str = "auto") -> str:
    device = preferred_device.strip().lower()
    if device not in _SUPPORTED_DEVICES:
        raise ValueError(f"unsupported torch device '{preferred_device}', expected one of: {sorted(_SUPPORTED_DEVICES)}")

    if device == "cpu":
        return "cpu"

    torch_module = _load_torch()
    if torch_module is None:
        if device == "auto":
            return "cpu"
        raise RuntimeError("torch is not available but a non-CPU device was requested")

    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(callable(getattr(cuda, "is_available", None)) and cuda.is_available())
    mps_available = _is_mps_available(torch_module)

    if device == "cuda":
        if not cuda_available:
            raise RuntimeError("torch device 'cuda' requested but CUDA is unavailable")
        return "cuda"

    if device == "mps":
        if not mps_available:
            raise RuntimeError("torch device 'mps' requested but MPS is unavailable")
        return "mps"

    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def resolve_cpu_runtime_device(preferred_device: str = "auto") -> str:
    device = preferred_device.strip().lower()
    if device not in _SUPPORTED_CPU_RUNTIME_DEVICES:
        raise ValueError(
            f"unsupported cpu runtime device '{preferred_device}', expected one of: {sorted(_SUPPORTED_CPU_RUNTIME_DEVICES)}"
        )

    if device == "cpu":
        return "cpu"
    if device == "mps":
        return resolve_torch_device("mps")

    torch_module = _load_torch()
    if torch_module is None:
        return "cpu"
    if _is_mps_available(torch_module):
        return "mps"
    return "cpu"
