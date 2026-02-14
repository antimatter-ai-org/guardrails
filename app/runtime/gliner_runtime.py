from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.runtime.torch_runtime import resolve_torch_device


class GlinerRuntime(ABC):
    @abstractmethod
    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        raise NotImplementedError


class LocalCpuGlinerRuntime(GlinerRuntime):
    def __init__(self, model_name: str, preferred_device: str = "auto") -> None:
        self.device = resolve_torch_device(preferred_device)
        self._model_name = model_name
        self._model: Any | None = None
        self._loading_started = False
        self._loading_lock = threading.Lock()
        self._load_error: str | None = None

    def _load_model(self) -> None:
        try:
            from gliner import GLiNER
        except Exception as exc:
            self._load_error = f"gliner import error: {exc}"
            return

        try:
            model = GLiNER.from_pretrained(self._model_name)
            if hasattr(model, "to"):
                model.to(self.device)
            self._model = model
        except Exception as exc:  # pragma: no cover - network/model availability dependent
            self._load_error = f"gliner model load error: {exc}"

    def _ensure_loading_started(self) -> None:
        with self._loading_lock:
            if self._loading_started:
                return
            self._loading_started = True
            thread = threading.Thread(target=self._load_model, daemon=True)
            thread.start()

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        if self._model is None:
            self._ensure_loading_started()
            return []
        return self._model.predict_entities(text, labels, threshold=threshold)


class PyTritonGlinerRuntime(GlinerRuntime):
    def __init__(
        self,
        model_name: str,
        pytriton_url: str,
        init_timeout_s: float,
        infer_timeout_s: float,
    ) -> None:
        self._model_name = model_name
        self._pytriton_url = pytriton_url
        self._init_timeout_s = init_timeout_s
        self._infer_timeout_s = infer_timeout_s
        self.device = "cuda"

    @staticmethod
    def _extract_detection_payload(output: Any) -> str:
        if isinstance(output, (bytes, bytearray)):
            return output.decode("utf-8")
        if isinstance(output, str):
            return output
        if isinstance(output, np.ndarray):
            if output.size == 0:
                return "[]"
            return PyTritonGlinerRuntime._extract_detection_payload(output.flat[0])
        return str(output)

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        try:
            from pytriton.client import ModelClient
        except Exception as exc:
            raise RuntimeError("PyTriton client is not installed. Install with guardrails-service[gpu].") from exc

        text_batch = np.array([[text.encode("utf-8")]], dtype=object)
        labels_batch = np.array([[json.dumps(labels).encode("utf-8")]], dtype=object)
        threshold_batch = np.array([[threshold]], dtype=np.float32)

        with ModelClient(
            url=self._pytriton_url,
            model_name=self._model_name,
            init_timeout_s=self._init_timeout_s,
            inference_timeout_s=self._infer_timeout_s,
        ) as client:
            result = client.infer_batch(
                text=text_batch,
                labels_json=labels_batch,
                threshold=threshold_batch,
            )

        raw_payload = self._extract_detection_payload(result.get("detections_json", []))
        parsed = json.loads(raw_payload)
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]


def build_gliner_runtime(
    runtime_mode: str,
    model_name: str,
    cpu_device: str,
    pytriton_url: str,
    pytriton_model_name: str,
    pytriton_init_timeout_s: float,
    pytriton_infer_timeout_s: float,
) -> GlinerRuntime:
    mode = runtime_mode.strip().lower()

    if mode == "cpu":
        return LocalCpuGlinerRuntime(model_name=model_name, preferred_device=cpu_device)

    if mode == "gpu":
        return PyTritonGlinerRuntime(
            model_name=pytriton_model_name,
            pytriton_url=pytriton_url,
            init_timeout_s=pytriton_init_timeout_s,
            infer_timeout_s=pytriton_infer_timeout_s,
        )

    raise ValueError("unsupported runtime mode, expected 'cpu' or 'gpu'")
