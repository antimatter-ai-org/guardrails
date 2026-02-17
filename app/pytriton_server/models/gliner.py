from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.pytriton_server.models.base import TritonModelBinding
from app.runtime.tokenizer_chunking import (
    chunk_text,
    deterministic_overlap_tokens,
)
from app.runtime.torch_runtime import resolve_torch_device


class GlinerTritonModel:
    def __init__(
        self,
        triton_model_name: str,
        gliner_hf_model_name: str,
        device: str,
        max_batch_size: int = 32,
    ) -> None:
        self._triton_model_name = triton_model_name
        self._gliner_hf_model_name = gliner_hf_model_name
        self._device = resolve_torch_device(device)
        self._max_batch_size = max_batch_size
        self._encoder_tokenizer: Any | None = None
        self._max_input_tokens: int | None = None
        self._overlap_tokens: int | None = None

        try:
            from gliner import GLiNER
        except Exception as exc:
            raise RuntimeError("gliner is not installed in pytriton server image") from exc

        self._model = GLiNER.from_pretrained(self._gliner_hf_model_name)
        if hasattr(self._model, "to"):
            self._model.to(self._device)
        self._load_chunking_limits()

    def _load_chunking_limits(self) -> None:
        gliner_cfg: dict[str, Any] | None = None
        base = Path(str(self._gliner_hf_model_name))
        if base.exists() and base.is_dir():
            cfg_path = base / "gliner_config.json"
            if cfg_path.exists():
                try:
                    gliner_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                except Exception:
                    gliner_cfg = None
        if gliner_cfg is None:
            cfg = getattr(self._model, "config", None)
            if isinstance(cfg, dict):
                gliner_cfg = dict(cfg)

        if not isinstance(gliner_cfg, dict):
            raise RuntimeError("unable to read GLiNER config for safe chunking (refuse to risk silent truncation)")
        encoder_name = str(gliner_cfg.get("model_name") or "").strip()
        if not encoder_name:
            raise RuntimeError("unable to determine GLiNER encoder model_name (refuse to risk silent truncation)")
        raw_max_len = gliner_cfg.get("max_len")
        try:
            max_len = int(raw_max_len)
        except Exception as exc:
            raise RuntimeError("unable to determine GLiNER max_len (refuse to risk silent truncation)") from exc
        if max_len < 2:
            raise RuntimeError("invalid GLiNER max_len (refuse to risk silent truncation)")
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError("transformers is required for safe chunking") from exc
        encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
        if not bool(getattr(encoder_tokenizer, "is_fast", False)):
            raise RuntimeError("fast tokenizer is required for safe chunking (offsets mapping)")
        specials = int(getattr(encoder_tokenizer, "num_special_tokens_to_add", lambda **_: 0)(pair=False))
        cap = int(max_len) - max(0, specials)
        if cap < 2:
            raise RuntimeError("invalid derived GLiNER context length")
        self._encoder_tokenizer = encoder_tokenizer
        self._max_input_tokens = int(cap)
        self._overlap_tokens = deterministic_overlap_tokens(int(cap))

    @staticmethod
    def _decode_cell(cell: object) -> str:
        if isinstance(cell, (bytes, bytearray)):
            return cell.decode("utf-8")
        if isinstance(cell, str):
            return cell
        return str(cell)

    def _infer_impl(self, text: np.ndarray, labels_json: np.ndarray, threshold: np.ndarray) -> dict[str, np.ndarray]:
        batch_size = text.shape[0]
        outputs: list[list[bytes]] = []

        for idx in range(batch_size):
            raw_text = self._decode_cell(text[idx][0])
            raw_labels = self._decode_cell(labels_json[idx][0])
            raw_threshold = float(threshold[idx][0])

            labels = json.loads(raw_labels)
            if not isinstance(labels, list):
                labels = []
            labels = [str(item) for item in labels]

            if self._encoder_tokenizer is None or self._max_input_tokens is None or self._overlap_tokens is None:
                raise RuntimeError("GLiNER tokenizer chunking is not initialized")
            windows = chunk_text(
                text=raw_text,
                tokenizer=self._encoder_tokenizer,
                max_input_tokens=self._max_input_tokens,
                overlap_tokens=self._overlap_tokens,
            )
            chunk_texts = [raw_text[w.text_start : w.text_end] for w in windows]
            # GLiNER batching support varies; keep it simple per chunk.
            dedup: dict[tuple[int, int, str], dict[str, Any]] = {}
            for window, chunk in zip(windows, chunk_texts, strict=False):
                preds = self._model.predict_entities(chunk, labels, threshold=raw_threshold)
                if not isinstance(preds, list):
                    continue
                for item in preds:
                    if not isinstance(item, dict):
                        continue
                    start = int(item.get("start", -1))
                    end = int(item.get("end", -1))
                    if end <= start:
                        continue
                    label = str(item.get("label", "")).strip()
                    if not label:
                        continue
                    score = float(item.get("score", raw_threshold))
                    global_start = int(window.text_start) + start
                    global_end = int(window.text_start) + end
                    key = (global_start, global_end, label)
                    existing = dedup.get(key)
                    if existing is None or score > float(existing.get("score", 0.0)):
                        dedup[key] = {
                            "start": global_start,
                            "end": global_end,
                            "text": raw_text[global_start:global_end],
                            "label": label,
                            "score": score,
                        }
            predictions = sorted(dedup.values(), key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))
            outputs.append([json.dumps(predictions, ensure_ascii=False).encode("utf-8")])

        return {"detections_json": np.array(outputs, dtype=object)}

    def binding(self) -> TritonModelBinding:
        from pytriton.decorators import batch
        from pytriton.model_config import ModelConfig, Tensor

        @batch
        def infer_func(**kwargs):
            return self._infer_impl(
                text=kwargs["text"],
                labels_json=kwargs["labels_json"],
                threshold=kwargs["threshold"],
            )

        return TritonModelBinding(
            name=self._triton_model_name,
            infer_func=infer_func,
            inputs=[
                Tensor(name="text", dtype=bytes, shape=(1,)),
                Tensor(name="labels_json", dtype=bytes, shape=(1,)),
                Tensor(name="threshold", dtype=np.float32, shape=(1,)),
            ],
            outputs=[
                Tensor(name="detections_json", dtype=bytes, shape=(1,)),
            ],
            config=ModelConfig(max_batch_size=self._max_batch_size),
        )
