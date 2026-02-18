from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class TritonTensorContract:
    name: str
    data_type: str


@dataclass(frozen=True, slots=True)
class TritonModelContract:
    name: str
    inputs: tuple[TritonTensorContract, ...]
    outputs: tuple[TritonTensorContract, ...]


_NP_DTYPE_TO_TRITON: dict[np.dtype[Any], str] = {
    np.dtype(np.bool_): "TYPE_BOOL",
    np.dtype(np.int8): "TYPE_INT8",
    np.dtype(np.int16): "TYPE_INT16",
    np.dtype(np.int32): "TYPE_INT32",
    np.dtype(np.int64): "TYPE_INT64",
    np.dtype(np.uint8): "TYPE_UINT8",
    np.dtype(np.uint16): "TYPE_UINT16",
    np.dtype(np.uint32): "TYPE_UINT32",
    np.dtype(np.uint64): "TYPE_UINT64",
    np.dtype(np.float16): "TYPE_FP16",
    np.dtype(np.float32): "TYPE_FP32",
    np.dtype(np.float64): "TYPE_FP64",
}


def parse_pytriton_url(url: str) -> tuple[str, int]:
    raw = url.strip()
    if not raw:
        raise ValueError("GR_PYTRITON_URL must not be empty")
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    raw = raw.split("/", 1)[0].strip()
    if not raw:
        raise ValueError(f"invalid GR_PYTRITON_URL: {url!r}")
    if ":" not in raw:
        return raw, 8000
    host, port_raw = raw.rsplit(":", 1)
    if not host or not port_raw:
        raise ValueError(f"invalid GR_PYTRITON_URL: {url!r}")
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ValueError(f"invalid GR_PYTRITON_URL port: {port_raw!r}") from exc
    return host, port


def triton_data_type_for_dtype(dtype: Any) -> str:
    if dtype in {bytes, bytearray, str, object, np.bytes_, np.object_}:
        return "TYPE_STRING"
    try:
        np_dtype = np.dtype(dtype)
    except Exception as exc:
        raise ValueError(f"unsupported tensor dtype: {dtype!r}") from exc
    mapped = _NP_DTYPE_TO_TRITON.get(np_dtype)
    if mapped is None:
        raise ValueError(f"unsupported tensor dtype: {dtype!r}")
    return mapped


def contract_from_binding(binding: Any) -> TritonModelContract:
    def _collect(tensors: list[Any]) -> tuple[TritonTensorContract, ...]:
        contracts: list[TritonTensorContract] = []
        for tensor in tensors:
            name = str(getattr(tensor, "name", "")).strip()
            if not name:
                raise ValueError(f"tensor has empty name in model {getattr(binding, 'name', '<unknown>')!r}")
            dtype = triton_data_type_for_dtype(getattr(tensor, "dtype", None))
            contracts.append(TritonTensorContract(name=name, data_type=dtype))
        return tuple(contracts)

    model_name = str(getattr(binding, "name", "")).strip()
    if not model_name:
        raise ValueError("binding name must not be empty")
    return TritonModelContract(
        name=model_name,
        inputs=_collect(list(getattr(binding, "inputs", []))),
        outputs=_collect(list(getattr(binding, "outputs", []))),
    )


def _http_get(base_url: str, path: str, timeout_s: float) -> tuple[int, bytes]:
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url, path),
        method="GET",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=max(0.1, timeout_s)) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read() or b""


def _fetch_model_config(base_url: str, model_name: str, timeout_s: float) -> dict[str, Any]:
    safe_name = urllib.parse.quote(model_name, safe="")
    status, body = _http_get(base_url, f"/v2/models/{safe_name}/config", timeout_s)
    if status != 200:
        raise RuntimeError(f"model config endpoint returned HTTP {status} for model {model_name!r}")
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse model config for {model_name!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"model config payload is not an object for model {model_name!r}")
    config = payload.get("config", payload)
    if not isinstance(config, dict):
        raise RuntimeError(f"model config payload missing config object for model {model_name!r}")
    return config


def _validate_contract(config: dict[str, Any], contract: TritonModelContract) -> str | None:
    def _collect(section: str) -> dict[str, str] | None:
        raw = config.get(section)
        if not isinstance(raw, list):
            return None
        collected: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                return None
            name = str(item.get("name", "")).strip()
            data_type = str(item.get("data_type", "")).strip().upper()
            if not name or not data_type:
                return None
            collected[name] = data_type
        return collected

    actual_inputs = _collect("input")
    actual_outputs = _collect("output")
    if actual_inputs is None or actual_outputs is None:
        return "invalid model config payload (input/output)"

    expected_inputs = {item.name: item.data_type for item in contract.inputs}
    expected_outputs = {item.name: item.data_type for item in contract.outputs}

    if set(actual_inputs.keys()) != set(expected_inputs.keys()):
        return f"input tensor names mismatch (expected={sorted(expected_inputs.keys())}, actual={sorted(actual_inputs.keys())})"
    if set(actual_outputs.keys()) != set(expected_outputs.keys()):
        return f"output tensor names mismatch (expected={sorted(expected_outputs.keys())}, actual={sorted(actual_outputs.keys())})"

    for name, expected_type in expected_inputs.items():
        if actual_inputs.get(name) != expected_type:
            return f"input tensor dtype mismatch for {name!r} (expected={expected_type}, actual={actual_inputs.get(name)})"
    for name, expected_type in expected_outputs.items():
        if actual_outputs.get(name) != expected_type:
            return f"output tensor dtype mismatch for {name!r} (expected={expected_type}, actual={actual_outputs.get(name)})"
    return None


def wait_for_triton_ready(
    *,
    pytriton_url: str,
    contracts: list[TritonModelContract],
    timeout_s: float | None,
    poll_interval_s: float = 0.5,
) -> None:
    host, port = parse_pytriton_url(pytriton_url)
    base_url = f"http://{host}:{port}"
    # IMPORTANT: do not impose a global readiness wall-clock cap. Readiness is a
    # startup concern; request handling fails closed immediately when models are
    # not ready, but the process should not "give up" based on elapsed time.
    http_timeout_s = 2.0
    if timeout_s is not None:
        # Backward-compatible knob: per-request socket timeout only (not a global deadline).
        try:
            http_timeout_s = max(0.1, float(timeout_s))
        except Exception:
            http_timeout_s = 2.0
    last_error = "unknown startup error"

    while True:
        try:
            live_status, _ = _http_get(base_url, "/v2/health/live", http_timeout_s)
        except Exception as exc:
            last_error = f"Triton live probe failed: {exc}"
            time.sleep(poll_interval_s)
            continue
        if live_status != 200:
            last_error = f"Triton live probe returned HTTP {live_status}"
            time.sleep(poll_interval_s)
            continue
        try:
            ready_status, _ = _http_get(base_url, "/v2/health/ready", http_timeout_s)
        except Exception as exc:
            last_error = f"Triton ready probe failed: {exc}"
            time.sleep(poll_interval_s)
            continue
        if ready_status != 200:
            last_error = f"Triton ready probe returned HTTP {ready_status}"
            time.sleep(poll_interval_s)
            continue

        model_error: str | None = None
        for contract in contracts:
            safe_name = urllib.parse.quote(contract.name, safe="")
            try:
                model_ready_status, _ = _http_get(base_url, f"/v2/models/{safe_name}/ready", http_timeout_s)
            except Exception as exc:
                model_error = f"model {contract.name!r} ready probe failed: {exc}"
                break
            if model_ready_status != 200:
                model_error = f"model {contract.name!r} ready probe returned HTTP {model_ready_status}"
                break
            try:
                config = _fetch_model_config(base_url, contract.name, http_timeout_s)
            except Exception as exc:
                model_error = str(exc)
                break
            validation_error = _validate_contract(config, contract)
            if validation_error is not None:
                # Deterministic misconfiguration: fail fast rather than polling forever.
                raise RuntimeError(f"model {contract.name!r} contract validation failed: {validation_error}")
        if model_error is None:
            return
        last_error = model_error
        time.sleep(poll_interval_s)
