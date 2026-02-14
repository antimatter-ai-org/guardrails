from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.runtime.gliner_chunking import GlinerChunkingConfig, run_chunked_inference
from app.runtime.torch_runtime import resolve_cpu_runtime_device


class GlinerRuntime(ABC):
    @abstractmethod
    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        raise NotImplementedError


class LocalCpuGlinerRuntime(GlinerRuntime):
    def __init__(
        self,
        model_name: str,
        preferred_device: str = "auto",
        chunking: GlinerChunkingConfig | None = None,
    ) -> None:
        self.device = resolve_cpu_runtime_device(preferred_device)
        self._model_name = model_name
        self._chunking = (chunking or GlinerChunkingConfig()).normalized()
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
        return run_chunked_inference(
            text=text,
            labels=labels,
            threshold=threshold,
            chunking=self._chunking,
            predict_batch=self._predict_batch,
        )

    def _predict_batch(self, texts: list[str], labels: list[str], threshold: float) -> list[list[dict[str, Any]]]:
        if self._model is None:
            return [[] for _ in texts]

        if hasattr(self._model, "inference"):
            try:
                batch_size = max(1, min(32, len(texts)))
                outputs = self._model.inference(
                    texts,
                    labels=labels,
                    threshold=threshold,
                    flat_ner=True,
                    batch_size=batch_size,
                )
                if isinstance(outputs, list):
                    return [list(item) if isinstance(item, list) else [] for item in outputs]
            except Exception:
                # Fallback to per-text prediction for maximum compatibility.
                pass

        results: list[list[dict[str, Any]]] = []
        for text in texts:
            outputs = self._model.predict_entities(text, labels, threshold=threshold)
            results.append(list(outputs) if isinstance(outputs, list) else [])
        return results


class PyTritonGlinerRuntime(GlinerRuntime):
    def __init__(
        self,
        model_name: str,
        pytriton_url: str,
        init_timeout_s: float,
        infer_timeout_s: float,
        chunking: GlinerChunkingConfig | None = None,
    ) -> None:
        self._model_name = model_name
        self._pytriton_url = pytriton_url
        self._init_timeout_s = init_timeout_s
        self._infer_timeout_s = infer_timeout_s
        self._chunking = (chunking or GlinerChunkingConfig()).normalized()
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
        return run_chunked_inference(
            text=text,
            labels=labels,
            threshold=threshold,
            chunking=self._chunking,
            predict_batch=self._predict_batch,
        )

    def _predict_batch(self, texts: list[str], labels: list[str], threshold: float) -> list[list[dict[str, Any]]]:
        try:
            from pytriton.client import ModelClient
        except Exception as exc:
            raise RuntimeError("PyTriton client is not installed. Install with guardrails-service[gpu].") from exc

        if not texts:
            return []

        text_batch = np.array([[text.encode("utf-8")] for text in texts], dtype=object)
        labels_payload = json.dumps(labels).encode("utf-8")
        labels_batch = np.array([[labels_payload] for _ in texts], dtype=object)
        threshold_batch = np.array([[threshold] for _ in texts], dtype=np.float32)

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

        payload = result.get("detections_json", [])
        parsed_batch: list[list[dict[str, Any]]] = []
        if isinstance(payload, np.ndarray):
            rows = list(payload)
        else:
            rows = [payload]

        for row in rows:
            cell = row[0] if isinstance(row, (list, tuple, np.ndarray)) and len(row) > 0 else row
            raw_payload = self._extract_detection_payload(cell)
            parsed = json.loads(raw_payload)
            if not isinstance(parsed, list):
                parsed_batch.append([])
                continue
            parsed_batch.append([item for item in parsed if isinstance(item, dict)])
        return parsed_batch


def build_gliner_runtime(
    runtime_mode: str,
    model_name: str,
    cpu_device: str,
    pytriton_url: str,
    pytriton_model_name: str,
    pytriton_init_timeout_s: float,
    pytriton_infer_timeout_s: float,
    chunking_enabled: bool = True,
    chunking_max_tokens: int = 320,
    chunking_overlap_tokens: int = 64,
    chunking_max_chunks: int = 64,
    chunking_boundary_lookback_tokens: int = 24,
) -> GlinerRuntime:
    mode = runtime_mode.strip().lower()
    if mode == "gpu":
        mode = "cuda"
    chunking = GlinerChunkingConfig(
        enabled=chunking_enabled,
        max_tokens=chunking_max_tokens,
        overlap_tokens=chunking_overlap_tokens,
        max_chunks=chunking_max_chunks,
        boundary_lookback_tokens=chunking_boundary_lookback_tokens,
    ).normalized()

    if mode == "cpu":
        return LocalCpuGlinerRuntime(
            model_name=model_name,
            preferred_device=cpu_device,
            chunking=chunking,
        )

    if mode == "cuda":
        return PyTritonGlinerRuntime(
            model_name=pytriton_model_name,
            pytriton_url=pytriton_url,
            init_timeout_s=pytriton_init_timeout_s,
            infer_timeout_s=pytriton_infer_timeout_s,
            chunking=chunking,
        )

    raise ValueError("unsupported runtime mode, expected 'cpu' or 'cuda'")
