from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.runtime import triton_readiness


def test_contract_from_binding_maps_tensor_dtypes() -> None:
    binding = SimpleNamespace(
        name="gliner",
        inputs=[
            SimpleNamespace(name="text", dtype=bytes),
            SimpleNamespace(name="threshold", dtype=np.float32),
        ],
        outputs=[SimpleNamespace(name="detections_json", dtype=bytes)],
    )

    contract = triton_readiness.contract_from_binding(binding)

    assert contract.name == "gliner"
    assert contract.inputs[0] == triton_readiness.TritonTensorContract(name="text", data_type="TYPE_STRING")
    assert contract.inputs[1] == triton_readiness.TritonTensorContract(name="threshold", data_type="TYPE_FP32")
    assert contract.outputs[0] == triton_readiness.TritonTensorContract(name="detections_json", data_type="TYPE_STRING")


def test_wait_for_triton_ready_succeeds_on_matching_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = triton_readiness.TritonModelContract(
        name="gliner",
        inputs=(
            triton_readiness.TritonTensorContract(name="text", data_type="TYPE_STRING"),
            triton_readiness.TritonTensorContract(name="labels_json", data_type="TYPE_STRING"),
            triton_readiness.TritonTensorContract(name="threshold", data_type="TYPE_FP32"),
        ),
        outputs=(triton_readiness.TritonTensorContract(name="detections_json", data_type="TYPE_STRING"),),
    )

    monkeypatch.setattr(triton_readiness, "_http_get", lambda *_args, **_kwargs: (200, b"{}"))
    monkeypatch.setattr(
        triton_readiness,
        "_fetch_model_config",
        lambda *_args, **_kwargs: {
            "input": [
                {"name": "text", "data_type": "TYPE_STRING"},
                {"name": "labels_json", "data_type": "TYPE_STRING"},
                {"name": "threshold", "data_type": "TYPE_FP32"},
            ],
            "output": [{"name": "detections_json", "data_type": "TYPE_STRING"}],
        },
    )

    triton_readiness.wait_for_triton_ready(
        pytriton_url="127.0.0.1:8000",
        contracts=[contract],
        timeout_s=0.5,
    )


def test_wait_for_triton_ready_fails_on_contract_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = triton_readiness.TritonModelContract(
        name="nemotron",
        inputs=(triton_readiness.TritonTensorContract(name="threshold", data_type="TYPE_FP32"),),
        outputs=(triton_readiness.TritonTensorContract(name="detections_json", data_type="TYPE_STRING"),),
    )
    values = [0.0, 0.05, 0.1, 0.15, 0.2]
    state = {"idx": 0}

    def fake_monotonic() -> float:
        idx = state["idx"]
        if idx >= len(values):
            return values[-1]
        state["idx"] = idx + 1
        return values[idx]

    monkeypatch.setattr(triton_readiness.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(triton_readiness.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(triton_readiness, "_http_get", lambda *_args, **_kwargs: (200, b"{}"))
    monkeypatch.setattr(
        triton_readiness,
        "_fetch_model_config",
        lambda *_args, **_kwargs: {
            "input": [{"name": "threshold", "data_type": "TYPE_INT32"}],
            "output": [{"name": "detections_json", "data_type": "TYPE_STRING"}],
        },
    )

    with pytest.raises(RuntimeError, match="contract validation failed"):
        triton_readiness.wait_for_triton_ready(
            pytriton_url="127.0.0.1:8000",
            contracts=[contract],
            timeout_s=0.1,
        )
