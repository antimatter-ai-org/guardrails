from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from app.pytriton_server.models.base import TritonModelBinding
from app.runtime.tokenizer_chunking import (
    chunk_text,
    deterministic_overlap_tokens,
    effective_max_tokens_for_token_classifier,
)
from app.runtime.torch_runtime import resolve_torch_device


def _normalize_label(label: str) -> str:
    value = label.strip()
    if not value:
        return ""
    return re.sub(r"^(?:B|I)-", "", value, flags=re.IGNORECASE).strip()


class TokenClassifierTritonModel:
    def __init__(
        self,
        *,
        triton_model_name: str,
        hf_model_name: str,
        device: str,
        aggregation_strategy: str = "simple",
        max_batch_size: int = 16,
    ) -> None:
        self._triton_model_name = triton_model_name
        self._hf_model_name = hf_model_name
        self._device = resolve_torch_device(device)
        self._aggregation_strategy = aggregation_strategy
        self._max_batch_size = max_batch_size

        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        except Exception as exc:
            raise RuntimeError("transformers runtime is not installed in pytriton server image") from exc

        if self._device == "cpu":
            pipeline_device: int | Any = -1
        else:
            pipeline_device = torch.device(self._device)

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._hf_model_name, use_fast=True, fix_mistral_regex=True)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(self._hf_model_name, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(self._hf_model_name)
        # Strict chunking: refuse to run without a fast tokenizer (offsets required).
        if not bool(getattr(tokenizer, "is_fast", False)):
            raise RuntimeError("fast tokenizer is required for safe chunking (offsets mapping)")
        self._tokenizer = tokenizer
        self._max_input_tokens = int(effective_max_tokens_for_token_classifier(model=model, tokenizer=tokenizer))
        self._overlap_tokens = deterministic_overlap_tokens(self._max_input_tokens)
        self._pipeline = pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy=self._aggregation_strategy,
            device=pipeline_device,
        )

    @staticmethod
    def _decode_cell(cell: object) -> str:
        if isinstance(cell, (bytes, bytearray)):
            return cell.decode("utf-8")
        if isinstance(cell, str):
            return cell
        return str(cell)

    @staticmethod
    def _normalize_outputs(raw_outputs: Any, expected_batch: int) -> list[list[dict[str, Any]]]:
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
                batches = [list(item) if isinstance(item, list) else [] for item in raw_outputs]
                if len(batches) < expected_batch:
                    batches.extend([[] for _ in range(expected_batch - len(batches))])
                return batches[:expected_batch]
        return [[] for _ in range(expected_batch)]

    def _infer_impl(self, text: np.ndarray, threshold: np.ndarray, labels_json: np.ndarray) -> dict[str, np.ndarray]:
        texts = [self._decode_cell(row[0]) for row in text]
        thresholds = [float(row[0]) for row in threshold]
        raw_labels = [self._decode_cell(row[0]) for row in labels_json]

        parsed_labels: list[set[str]] = []
        for payload in raw_labels:
            try:
                data = json.loads(payload)
            except Exception:
                data = []
            if not isinstance(data, list):
                data = []
            normalized = {_normalize_label(str(item)).lower() for item in data if str(item).strip()}
            parsed_labels.append(normalized)

        outputs: list[list[bytes]] = []
        for sample_text, sample_threshold, allowed_labels in zip(texts, thresholds, parsed_labels, strict=False):
            windows = chunk_text(
                text=sample_text,
                tokenizer=self._tokenizer,
                max_input_tokens=self._max_input_tokens,
                overlap_tokens=self._overlap_tokens,
            )
            chunk_texts = [sample_text[w.text_start : w.text_end] for w in windows]
            chunk_raw = self._pipeline(chunk_texts, batch_size=max(1, min(self._max_batch_size, len(chunk_texts))))
            chunk_batches = self._normalize_outputs(chunk_raw, expected_batch=len(chunk_texts))
            dedup: dict[tuple[int, int, str], dict[str, Any]] = {}
            for window, predictions in zip(windows, chunk_batches, strict=False):
                for item in predictions:
                    if not isinstance(item, dict):
                        continue
                    start = int(item.get("start", -1))
                    end = int(item.get("end", -1))
                    if end <= start:
                        continue
                    score = float(item.get("score", 0.0))
                    if score < sample_threshold:
                        continue
                    label = _normalize_label(str(item.get("entity_group", item.get("entity", ""))))
                    if not label:
                        continue
                    if allowed_labels and label.lower() not in allowed_labels:
                        continue
                    global_start = int(window.text_start) + start
                    global_end = int(window.text_start) + end
                    key = (global_start, global_end, label)
                    existing = dedup.get(key)
                    if existing is None or score > float(existing.get("score", 0.0)):
                        dedup[key] = {
                            "start": global_start,
                            "end": global_end,
                            "text": sample_text[global_start:global_end],
                            "label": label,
                            "score": score,
                        }
            sample_output = sorted(dedup.values(), key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))
            outputs.append([json.dumps(sample_output, ensure_ascii=False).encode("utf-8")])

        return {"detections_json": np.array(outputs, dtype=object)}

    def binding(self) -> TritonModelBinding:
        from pytriton.decorators import batch
        from pytriton.model_config import ModelConfig, Tensor

        @batch
        def infer_func(**kwargs):
            return self._infer_impl(
                text=kwargs["text"],
                threshold=kwargs["threshold"],
                labels_json=kwargs["labels_json"],
            )

        return TritonModelBinding(
            name=self._triton_model_name,
            infer_func=infer_func,
            inputs=[
                Tensor(name="text", dtype=bytes, shape=(1,)),
                Tensor(name="threshold", dtype=np.float32, shape=(1,)),
                Tensor(name="labels_json", dtype=bytes, shape=(1,)),
            ],
            outputs=[Tensor(name="detections_json", dtype=bytes, shape=(1,))],
            config=ModelConfig(max_batch_size=self._max_batch_size),
        )
