from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.runtime.tokenizer_chunking import (
    TextChunkWindow,
    chunk_text,
    deterministic_overlap_tokens,
    effective_max_tokens_for_token_classifier,
)
from app.runtime.triton_readiness import (
    TritonModelContract,
    TritonTensorContract,
    wait_for_triton_ready,
)
from app.runtime.torch_runtime import resolve_cpu_runtime_device


class TokenClassifierRuntime(ABC):
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


def _normalize_token_classifier_label(label: str) -> str:
    raw = label.strip()
    if not raw:
        return ""
    normalized = re.sub(r"^(?:B|I)-", "", raw, flags=re.IGNORECASE)
    return normalized.strip()


class LocalCpuTokenClassifierRuntime(TokenClassifierRuntime):
    def __init__(
        self,
        *,
        model_name: str,
        preferred_device: str = "auto",
        aggregation_strategy: str = "simple",
    ) -> None:
        self.device = resolve_cpu_runtime_device(preferred_device)
        self._model_name = model_name
        self._aggregation_strategy = aggregation_strategy
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._max_input_tokens: int | None = None
        self._overlap_tokens: int | None = None
        self._pipeline: Any | None = None
        self._load_error: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        except Exception as exc:
            self._load_error = f"transformers import error: {exc}"
            return

        if self.device == "cpu":
            pipeline_device: int | Any = -1
        else:
            pipeline_device = torch.device(self.device)

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name, use_fast=True)
            model = AutoModelForTokenClassification.from_pretrained(self._model_name)
            max_input_tokens = effective_max_tokens_for_token_classifier(model=model, tokenizer=tokenizer)
            self._pipeline = pipeline(
                task="token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy=self._aggregation_strategy,
                device=pipeline_device,
            )
            self._tokenizer = tokenizer
            self._model = model
            self._max_input_tokens = int(max_input_tokens)
            self._overlap_tokens = deterministic_overlap_tokens(int(max_input_tokens))
        except Exception as exc:  # pragma: no cover - model availability dependent
            self._load_error = f"token-classifier load error: {exc}"

    @staticmethod
    def _to_sample_batches(raw_outputs: Any, expected_batch: int) -> list[list[dict[str, Any]]]:
        if expected_batch <= 0:
            return []
        if raw_outputs is None:
            return [[] for _ in range(expected_batch)]
        if isinstance(raw_outputs, list):
            if not raw_outputs:
                return [[] for _ in range(expected_batch)]
            if all(isinstance(item, dict) for item in raw_outputs):
                return [list(raw_outputs)] + [[] for _ in range(max(0, expected_batch - 1))]
            if all(isinstance(item, list) for item in raw_outputs):
                casted = [list(item) if isinstance(item, list) else [] for item in raw_outputs]
                if len(casted) < expected_batch:
                    casted.extend([[] for _ in range(expected_batch - len(casted))])
                return casted[:expected_batch]
        return [[] for _ in range(expected_batch)]

    def _predict_batch(self, texts: list[str], labels: list[str], threshold: float) -> list[list[dict[str, Any]]]:
        if not texts:
            return []
        if self._pipeline is None:
            return [[] for _ in texts]

        allowed_labels = {_normalize_token_classifier_label(item).lower() for item in labels if item.strip()}
        raw_outputs = self._pipeline(texts, batch_size=max(1, min(16, len(texts))))
        batches = self._to_sample_batches(raw_outputs, expected_batch=len(texts))

        output: list[list[dict[str, Any]]] = []
        for text, predictions in zip(texts, batches, strict=False):
            sample_results: list[dict[str, Any]] = []
            for item in predictions:
                if not isinstance(item, dict):
                    continue
                start = int(item.get("start", -1))
                end = int(item.get("end", -1))
                if end <= start:
                    continue
                score = float(item.get("score", 0.0))
                if score < threshold:
                    continue
                label = str(item.get("entity_group", item.get("entity", ""))).strip()
                label = _normalize_token_classifier_label(label)
                if not label:
                    continue
                if allowed_labels and label.lower() not in allowed_labels:
                    continue
                sample_results.append(
                    {
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                        "label": label,
                        "score": score,
                    }
                )
            output.append(sample_results)
        return output

    @staticmethod
    def _merge_window_predictions(
        *,
        text: str,
        windows: Sequence[TextChunkWindow],
        window_predictions: Sequence[Sequence[dict[str, Any]]],
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
        if self._pipeline is None or self._tokenizer is None or self._max_input_tokens is None or self._overlap_tokens is None:
            raise RuntimeError(self._load_error or "token-classifier runtime is not ready")
        windows = chunk_text(
            text=text,
            tokenizer=self._tokenizer,
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
        return self._pipeline is not None

    def is_ready(self) -> bool:
        return self._pipeline is not None

    def load_error(self) -> str | None:
        return self._load_error


class PyTritonTokenClassifierRuntime(TokenClassifierRuntime):
    def __init__(
        self,
        *,
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
        self._tokenizer: Any | None = None
        self._max_input_tokens: int | None = None
        self._overlap_tokens: int | None = None
        self._contract = TritonModelContract(
            name=triton_model_name,
            inputs=(
                TritonTensorContract(name="text", data_type="TYPE_STRING"),
                TritonTensorContract(name="threshold", data_type="TYPE_FP32"),
                TritonTensorContract(name="labels_json", data_type="TYPE_STRING"),
            ),
            outputs=(TritonTensorContract(name="detections_json", data_type="TYPE_STRING"),),
        )

    def _load_tokenizer(self) -> None:
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except Exception as exc:
            self._load_error = f"transformers import error: {exc}"
            return
        try:
            tokenizer = AutoTokenizer.from_pretrained(self._hf_model_name, use_fast=True)
            model = AutoModelForTokenClassification.from_pretrained(self._hf_model_name)
            max_input_tokens = effective_max_tokens_for_token_classifier(model=model, tokenizer=tokenizer)
            self._tokenizer = tokenizer
            self._max_input_tokens = int(max_input_tokens)
            self._overlap_tokens = deterministic_overlap_tokens(int(max_input_tokens))
        except Exception as exc:  # pragma: no cover
            self._load_error = f"token-classifier tokenizer load error: {exc}"

    @staticmethod
    def _extract_payload(output: Any) -> str:
        if isinstance(output, (bytes, bytearray)):
            return output.decode("utf-8")
        if isinstance(output, str):
            return output
        if isinstance(output, np.ndarray):
            if output.size == 0:
                return "[]"
            return PyTritonTokenClassifierRuntime._extract_payload(output.flat[0])
        return str(output)

    @staticmethod
    def _extract_server_max_batch_size(error: Exception) -> int | None:
        message = str(error)
        match = re.search(r"batch-size must be <=\s*(\d+)", message, flags=re.IGNORECASE)
        if not match:
            return None
        value = int(match.group(1))
        if value < 1:
            return None
        return value

    @staticmethod
    def _parse_payload(payload: Any) -> list[list[dict[str, Any]]]:
        parsed_batch: list[list[dict[str, Any]]] = []
        if isinstance(payload, np.ndarray):
            rows = list(payload)
        else:
            rows = [payload]

        for row in rows:
            cell = row[0] if isinstance(row, (list, tuple, np.ndarray)) and len(row) > 0 else row
            raw_payload = PyTritonTokenClassifierRuntime._extract_payload(cell)
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
                threshold_batch = np.array([[threshold] for _ in chunk], dtype=np.float32)
                labels_batch = np.array([[labels_payload] for _ in chunk], dtype=object)
                try:
                    result = client.infer_batch(
                        text=text_batch,
                        threshold=threshold_batch,
                        labels_json=labels_batch,
                    )
                except Exception as exc:
                    server_limit = self._extract_server_max_batch_size(exc)
                    if server_limit is not None and server_limit < max_batch_size:
                        max_batch_size = server_limit
                        self._max_batch_size_hint = server_limit
                        continue
                    raise
                outputs.extend(self._parse_payload(result.get("detections_json", [])))
                idx = end

        return outputs

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        if not self._ready:
            raise RuntimeError(self._load_error or "pytriton token-classifier runtime is not ready")
        if self._tokenizer is None or self._max_input_tokens is None or self._overlap_tokens is None:
            self._load_tokenizer()
        if self._tokenizer is None or self._max_input_tokens is None or self._overlap_tokens is None:
            raise RuntimeError(self._load_error or "pytriton token-classifier runtime is not ready")
        windows = chunk_text(
            text=text,
            tokenizer=self._tokenizer,
            max_input_tokens=self._max_input_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        chunk_texts = [text[w.text_start : w.text_end] for w in windows]
        chunk_predictions = self._predict_batch(chunk_texts, labels, threshold)
        return LocalCpuTokenClassifierRuntime._merge_window_predictions(  # noqa: SLF001
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


def build_token_classifier_runtime(
    *,
    runtime_mode: str,
    model_name: str,
    cpu_device: str,
    pytriton_url: str,
    pytriton_model_name: str,
    pytriton_init_timeout_s: float,
    pytriton_infer_timeout_s: float,
    aggregation_strategy: str = "simple",
) -> TokenClassifierRuntime:
    mode = runtime_mode.strip().lower()

    if mode == "cpu":
        return LocalCpuTokenClassifierRuntime(
            model_name=model_name,
            preferred_device=cpu_device,
            aggregation_strategy=aggregation_strategy,
        )
    if mode == "cuda":
        return PyTritonTokenClassifierRuntime(
            triton_model_name=pytriton_model_name,
            hf_model_name=model_name,
            pytriton_url=pytriton_url,
            init_timeout_s=pytriton_init_timeout_s,
            infer_timeout_s=pytriton_infer_timeout_s,
        )
    raise ValueError(f"unsupported runtime_mode '{runtime_mode}', expected 'cpu' or 'cuda'")
