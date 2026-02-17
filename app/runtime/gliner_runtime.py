from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from app.runtime.tokenizer_chunking import (
    TextChunkWindow,
    chunk_text,
    deterministic_overlap_tokens,
)
from app.runtime.triton_readiness import (
    TritonModelContract,
    TritonTensorContract,
    wait_for_triton_ready,
)
from app.runtime.torch_runtime import resolve_cpu_runtime_device


class GlinerRuntime(ABC):
    @abstractmethod
    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def ensure_ready(self, timeout_s: float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_error(self) -> str | None:
        raise NotImplementedError


class LocalCpuGlinerRuntime(GlinerRuntime):
    def __init__(
        self,
        model_name: str,
        preferred_device: str = "auto",
    ) -> None:
        self.device = resolve_cpu_runtime_device(preferred_device)
        self._model_name = model_name
        self._model: Any | None = None
        self._encoder_tokenizer: Any | None = None
        self._max_input_tokens: int | None = None
        self._overlap_tokens: int | None = None
        self._load_error: str | None = None
        self._load_model()

    @staticmethod
    def _read_gliner_config(model_source: str) -> dict[str, Any] | None:
        try:
            base = Path(str(model_source))
        except Exception:
            return None
        if not base.exists() or not base.is_dir():
            return None
        cfg_path = base / "gliner_config.json"
        if not cfg_path.exists():
            return None
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _extract_gliner_config_from_model(model: Any) -> dict[str, Any] | None:
        cfg = getattr(model, "config", None)
        if isinstance(cfg, dict):
            return dict(cfg)
        return None

    @staticmethod
    def _derive_chunking_limits(*, gliner_cfg: dict[str, Any], encoder_tokenizer: Any) -> int:
        raw_max_len = gliner_cfg.get("max_len")
        try:
            max_len = int(raw_max_len)
        except Exception as exc:
            raise RuntimeError("unable to determine GLiNER max_len (refuse to risk silent truncation)") from exc
        if max_len < 2:
            raise RuntimeError("invalid GLiNER max_len (refuse to risk silent truncation)")
        specials = int(getattr(encoder_tokenizer, "num_special_tokens_to_add", lambda **_: 0)(pair=False))
        cap = max_len - max(0, specials)
        if cap < 2:
            raise RuntimeError("invalid derived GLiNER context length")
        return cap

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
            # Strict chunking: derive encoder tokenizer + max_len from GLiNER config.
            gliner_cfg = self._read_gliner_config(self._model_name) or self._extract_gliner_config_from_model(model)
            if not isinstance(gliner_cfg, dict):
                raise RuntimeError("unable to read GLiNER config (refuse to risk silent truncation)")
            encoder_name = str(gliner_cfg.get("model_name") or "").strip()
            if not encoder_name:
                raise RuntimeError("unable to determine GLiNER encoder model_name (refuse to risk silent truncation)")
            try:
                from transformers import AutoTokenizer
            except Exception as exc:
                raise RuntimeError("transformers is required for safe chunking") from exc
            encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
            max_input_tokens = self._derive_chunking_limits(gliner_cfg=gliner_cfg, encoder_tokenizer=encoder_tokenizer)
            self._encoder_tokenizer = encoder_tokenizer
            self._max_input_tokens = int(max_input_tokens)
            self._overlap_tokens = deterministic_overlap_tokens(int(max_input_tokens))
        except Exception as exc:  # pragma: no cover - network/model availability dependent
            self._load_error = f"gliner model load error: {exc}"

    @staticmethod
    def _merge_window_predictions(
        *,
        text: str,
        windows: list[TextChunkWindow],
        window_predictions: list[list[dict[str, Any]]],
        default_threshold: float,
    ) -> list[dict[str, Any]]:
        dedup: dict[tuple[int, int, str], dict[str, Any]] = {}
        text_len = len(text)
        for window, predictions in zip(windows, window_predictions, strict=False):
            for item in predictions:
                if not isinstance(item, dict):
                    continue
                local_start = int(item.get("start", -1))
                local_end = int(item.get("end", -1))
                if local_end <= local_start:
                    continue
                global_start = window.text_start + local_start
                global_end = window.text_start + local_end
                if global_start < 0 or global_end > text_len or global_end <= global_start:
                    continue
                label = str(item.get("label", "")).strip()
                if not label:
                    continue
                score = float(item.get("score", default_threshold))
                key = (global_start, global_end, label)
                existing = dedup.get(key)
                if existing is None or score > float(existing.get("score", 0.0)):
                    dedup[key] = {
                        "start": global_start,
                        "end": global_end,
                        "text": text[global_start:global_end],
                        "label": label,
                        "score": score,
                    }
        return sorted(dedup.values(), key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        if (
            self._model is None
            or self._encoder_tokenizer is None
            or self._max_input_tokens is None
            or self._overlap_tokens is None
        ):
            raise RuntimeError(self._load_error or "gliner runtime is not ready")
        windows = chunk_text(
            text=text,
            tokenizer=self._encoder_tokenizer,
            max_input_tokens=self._max_input_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        chunk_texts = [text[w.text_start : w.text_end] for w in windows]
        chunk_predictions = self._predict_batch(chunk_texts, labels, threshold)
        return self._merge_window_predictions(
            text=text,
            windows=windows,
            window_predictions=chunk_predictions,
            default_threshold=threshold,
        )

    def ensure_ready(self, timeout_s: float) -> bool:
        _ = timeout_s
        return self._model is not None

    def is_ready(self) -> bool:
        return self._model is not None

    def load_error(self) -> str | None:
        return self._load_error

    def _predict_batch(self, texts: list[str], labels: list[str], threshold: float) -> list[list[dict[str, Any]]]:
        if self._model is None:
            raise RuntimeError(self._load_error or "gliner runtime is not ready")

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
        triton_model_name: str,
        hf_model_name: str,
        pytriton_url: str,
        init_timeout_s: float,
        infer_timeout_s: float,
    ) -> None:
        self._model_name = triton_model_name
        self._hf_model_name = hf_model_name
        self._pytriton_url = pytriton_url
        self._init_timeout_s = max(float(init_timeout_s), float(infer_timeout_s))
        self._infer_timeout_s = infer_timeout_s
        self._max_batch_size_hint = 32
        self.device = "cuda"
        self._ready = False
        self._load_error: str | None = None
        self._encoder_tokenizer: Any | None = None
        self._max_input_tokens: int | None = None
        self._overlap_tokens: int | None = None
        self._contract = TritonModelContract(
            name=triton_model_name,
            inputs=(
                TritonTensorContract(name="text", data_type="TYPE_STRING"),
                TritonTensorContract(name="labels_json", data_type="TYPE_STRING"),
                TritonTensorContract(name="threshold", data_type="TYPE_FP32"),
            ),
            outputs=(TritonTensorContract(name="detections_json", data_type="TYPE_STRING"),),
        )

    def _load_tokenizer(self) -> None:
        # Strict chunking: require gliner_config.json to be present locally (model_dir mount)
        # or embedded in the GLiNER package config. In CUDA mode we avoid loading model weights.
        gliner_cfg = LocalCpuGlinerRuntime._read_gliner_config(self._hf_model_name)  # noqa: SLF001
        if not isinstance(gliner_cfg, dict):
            self._load_error = "unable to read GLiNER config for safe chunking (refuse to risk silent truncation)"
            return
        encoder_name = str(gliner_cfg.get("model_name") or "").strip()
        if not encoder_name:
            self._load_error = "unable to determine GLiNER encoder model_name (refuse to risk silent truncation)"
            return
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            self._load_error = f"transformers import error: {exc}"
            return
        try:
            encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
            max_input_tokens = LocalCpuGlinerRuntime._derive_chunking_limits(  # noqa: SLF001
                gliner_cfg=gliner_cfg,
                encoder_tokenizer=encoder_tokenizer,
            )
            self._encoder_tokenizer = encoder_tokenizer
            self._max_input_tokens = int(max_input_tokens)
            self._overlap_tokens = deterministic_overlap_tokens(int(max_input_tokens))
        except Exception as exc:  # pragma: no cover
            self._load_error = f"gliner tokenizer load error: {exc}"

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
        if not self._ready:
            raise RuntimeError(self._load_error or "pytriton gliner runtime is not ready")
        if self._encoder_tokenizer is None or self._max_input_tokens is None or self._overlap_tokens is None:
            self._load_tokenizer()
        if self._encoder_tokenizer is None or self._max_input_tokens is None or self._overlap_tokens is None:
            raise RuntimeError(self._load_error or "pytriton gliner runtime is not ready")
        windows = chunk_text(
            text=text,
            tokenizer=self._encoder_tokenizer,
            max_input_tokens=self._max_input_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        chunk_texts = [text[w.text_start : w.text_end] for w in windows]
        chunk_predictions = self._predict_batch(chunk_texts, labels, threshold)
        return LocalCpuGlinerRuntime._merge_window_predictions(  # noqa: SLF001
            text=text,
            windows=windows,
            window_predictions=chunk_predictions,
            default_threshold=threshold,
        )

    def ensure_ready(self, timeout_s: float) -> bool:
        if self._ready:
            return True
        try:
            wait_for_triton_ready(
                pytriton_url=self._pytriton_url,
                contracts=[self._contract],
                timeout_s=max(0.1, float(timeout_s)),
            )
            self._ready = True
            self._load_error = None
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def is_ready(self) -> bool:
        return self._ready

    def load_error(self) -> str | None:
        return self._load_error

    @staticmethod
    def _extract_server_max_batch_size(error: Exception) -> int | None:
        message = str(error)
        match = re.search(r"batch-size must be <=\s*(\d+)", message, re.IGNORECASE)
        if not match:
            return None
        value = int(match.group(1))
        if value < 1:
            return None
        return value

    @staticmethod
    def _parse_detections_payload(payload: Any) -> list[list[dict[str, Any]]]:
        parsed_batch: list[list[dict[str, Any]]] = []
        if isinstance(payload, np.ndarray):
            rows = list(payload)
        else:
            rows = [payload]

        for row in rows:
            cell = row[0] if isinstance(row, (list, tuple, np.ndarray)) and len(row) > 0 else row
            raw_payload = PyTritonGlinerRuntime._extract_detection_payload(cell)
            parsed = json.loads(raw_payload)
            if not isinstance(parsed, list):
                parsed_batch.append([])
                continue
            parsed_batch.append([item for item in parsed if isinstance(item, dict)])
        return parsed_batch

    def _predict_batch(self, texts: list[str], labels: list[str], threshold: float) -> list[list[dict[str, Any]]]:
        try:
            from pytriton.client import ModelClient
        except Exception as exc:
            raise RuntimeError("PyTriton client is not installed. Install with guardrails-service[cuda].") from exc

        if not texts:
            return []

        labels_payload = json.dumps(labels, ensure_ascii=False).encode("utf-8")
        outputs: list[list[dict[str, Any]]] = []
        max_batch_size = max(1, int(self._max_batch_size_hint))
        with ModelClient(
            url=self._pytriton_url,
            model_name=self._model_name,
            init_timeout_s=self._init_timeout_s,
            inference_timeout_s=self._infer_timeout_s,
        ) as client:
            idx = 0
            while idx < len(texts):
                end = min(idx + max_batch_size, len(texts))
                chunk = texts[idx:end]
                text_batch = np.array([[text.encode("utf-8")] for text in chunk], dtype=object)
                labels_batch = np.array([[labels_payload] for _ in chunk], dtype=object)
                threshold_batch = np.array([[threshold] for _ in chunk], dtype=np.float32)
                try:
                    result = client.infer_batch(
                        text=text_batch,
                        labels_json=labels_batch,
                        threshold=threshold_batch,
                    )
                except Exception as exc:
                    server_limit = self._extract_server_max_batch_size(exc)
                    if server_limit is not None and server_limit < max_batch_size:
                        max_batch_size = server_limit
                        self._max_batch_size_hint = server_limit
                        continue
                    raise
                outputs.extend(self._parse_detections_payload(result.get("detections_json", [])))
                idx = end
        return outputs


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
        return LocalCpuGlinerRuntime(
            model_name=model_name,
            preferred_device=cpu_device,
        )

    if mode == "cuda":
        return PyTritonGlinerRuntime(
            triton_model_name=pytriton_model_name,
            hf_model_name=model_name,
            pytriton_url=pytriton_url,
            init_timeout_s=pytriton_init_timeout_s,
            infer_timeout_s=pytriton_infer_timeout_s,
        )

    raise ValueError("unsupported runtime mode, expected 'cpu' or 'cuda'")
