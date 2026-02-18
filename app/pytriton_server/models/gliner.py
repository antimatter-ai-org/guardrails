from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from app.pytriton_server.models.base import TritonModelBinding
from app.runtime.gliner_word_chunking import (
    build_prompt_tokens_for_length_check,
    chunk_text_by_gliner_words,
    deterministic_overlap_words,
    gliner_prompt_len_words,
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
        self._encoder_max_len: int | None = None
        self._max_len_words: int | None = None
        self._max_width: int = 10
        self._ent_token: str = "<ENT>"
        self._sep_token: str = "<SEP>"

        try:
            from gliner import GLiNER
        except Exception as exc:
            raise RuntimeError("gliner is not installed in pytriton server image") from exc

        self._model = GLiNER.from_pretrained(self._gliner_hf_model_name)
        if hasattr(self._model, "to"):
            self._model.to(self._device)
        self._load_chunking_limits()

    @staticmethod
    def _reasonable_max_length(value: Any) -> int | None:
        try:
            as_int = int(value)
        except Exception:
            return None
        if as_int < 2:
            return None
        if as_int > 100_000:
            return None
        return as_int

    @classmethod
    def _resolve_encoder_max_len(cls, *, encoder_name: str, encoder_tokenizer: Any) -> int:
        max_pos: int | None = None
        try:
            from transformers import AutoConfig
        except Exception:
            AutoConfig = None  # type: ignore[assignment]
        if AutoConfig is not None:
            try:
                cfg = AutoConfig.from_pretrained(encoder_name)
                max_pos = cls._reasonable_max_length(getattr(cfg, "max_position_embeddings", None))
            except Exception:
                max_pos = None
        if max_pos is None:
            max_pos = cls._reasonable_max_length(getattr(encoder_tokenizer, "model_max_length", None))
        if max_pos is None:
            raise RuntimeError("unable to determine GLiNER encoder context length (refuse to risk silent truncation)")
        return int(max_pos)

    @staticmethod
    def _extract_truncation_warning(messages: list[warnings.WarningMessage]) -> str | None:
        for msg in messages:
            try:
                text = str(msg.message)
            except Exception:
                continue
            if re.search(r"Sentence of length .*truncated to", text, flags=re.IGNORECASE):
                return text
        return None

    def _prompt_str_for_length_check(self, *, labels: list[str]) -> str:
        return " ".join(
            build_prompt_tokens_for_length_check(
                ent_token=self._ent_token,
                sep_token=self._sep_token,
                labels=labels,
            )
        )

    def _would_tokenizer_truncate(self, *, prompt_str: str, chunk_text: str) -> bool:
        if self._encoder_tokenizer is None or self._encoder_max_len is None:
            raise RuntimeError("GLiNER tokenizer limits are not initialized")
        full = prompt_str if not chunk_text else (prompt_str + " " + chunk_text if prompt_str else chunk_text)
        enc = self._encoder_tokenizer(full, add_special_tokens=True, truncation=False)
        input_ids = enc.get("input_ids") if isinstance(enc, dict) else getattr(enc, "input_ids", None)
        if input_ids is None:
            raise RuntimeError("unable to inspect tokenizer output (refuse to risk silent truncation)")
        return len(input_ids) > int(self._encoder_max_len)

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
            elif hasattr(cfg, "to_dict"):
                try:
                    gliner_cfg = dict(cfg.to_dict())
                except Exception:
                    gliner_cfg = None
            elif hasattr(cfg, "__dict__"):
                try:
                    gliner_cfg = dict(getattr(cfg, "__dict__"))
                except Exception:
                    gliner_cfg = None
        if gliner_cfg is None:
            cfg_dict = getattr(self._model, "config_dict", None)
            if isinstance(cfg_dict, dict):
                gliner_cfg = dict(cfg_dict)

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
        self._max_len_words = int(max_len)
        raw_max_width = gliner_cfg.get("max_width", 10)
        try:
            self._max_width = int(raw_max_width)
        except Exception:
            self._max_width = 10
        self._ent_token = str(gliner_cfg.get("ent_token") or gliner_cfg.get("entity_token") or "<ENT>")
        self._sep_token = str(gliner_cfg.get("sep_token") or gliner_cfg.get("separator_token") or "<SEP>")
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError("transformers is required for safe chunking") from exc
        encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
        if not bool(getattr(encoder_tokenizer, "is_fast", False)):
            raise RuntimeError("fast tokenizer is required for safe chunking (offsets mapping)")
        self._encoder_max_len = self._resolve_encoder_max_len(encoder_name=encoder_name, encoder_tokenizer=encoder_tokenizer)
        self._encoder_tokenizer = encoder_tokenizer

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

            if self._encoder_tokenizer is None or self._encoder_max_len is None or self._max_len_words is None:
                raise RuntimeError("GLiNER tokenizer limits are not initialized")

            prompt_len = gliner_prompt_len_words(labels)
            max_text_words = int(self._max_len_words) - int(prompt_len)
            if max_text_words < 1:
                raise RuntimeError("GLiNER prompt consumes entire context window (refuse to risk truncation)")

            prompt_str = self._prompt_str_for_length_check(labels=labels)
            if self._would_tokenizer_truncate(prompt_str=prompt_str, chunk_text=""):
                raise RuntimeError("GLiNER prompt exceeds encoder context length (refuse to risk truncation)")

            attempt_words = int(max_text_words)
            while True:
                overlap_words = deterministic_overlap_words(max_len_words=attempt_words, max_width=int(self._max_width))
                windows = chunk_text_by_gliner_words(
                    text=raw_text,
                    max_text_words=attempt_words,
                    overlap_words=overlap_words,
                )
                chunk_texts = [raw_text[w.text_start : w.text_end] for w in windows]

                overflow = False
                for chunk in chunk_texts:
                    if self._would_tokenizer_truncate(prompt_str=prompt_str, chunk_text=chunk):
                        overflow = True
                        break
                if overflow:
                    next_words = attempt_words // 2
                    if next_words < 1:
                        raise RuntimeError("unable to build GLiNER chunks without tokenizer truncation")
                    attempt_words = next_words
                    continue

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    preds_by_chunk = self._infer_gliner_chunks(chunk_texts, labels=labels, threshold=raw_threshold)
                trunc_msg = self._extract_truncation_warning(list(caught))
                if trunc_msg is not None:
                    next_words = attempt_words // 2
                    if next_words < 1:
                        raise RuntimeError(f"GLiNER truncation could not be eliminated: {trunc_msg}")
                    attempt_words = next_words
                    continue
                break

            dedup: dict[tuple[int, int, str], dict[str, Any]] = {}
            for window, preds in zip(windows, preds_by_chunk, strict=False):
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

    def _infer_gliner_chunks(self, chunk_texts: list[str], *, labels: list[str], threshold: float) -> list[list[dict[str, Any]]]:
        # Prefer batched inference where available; fall back to per-text for compatibility.
        if hasattr(self._model, "inference"):
            try:
                batch_size = max(1, min(int(self._max_batch_size), len(chunk_texts)))
                outputs = self._model.inference(
                    chunk_texts,
                    labels=labels,
                    threshold=threshold,
                    flat_ner=True,
                    batch_size=batch_size,
                )
                if isinstance(outputs, list):
                    return [list(item) if isinstance(item, list) else [] for item in outputs]
            except Exception:
                pass
        results: list[list[dict[str, Any]]] = []
        for chunk in chunk_texts:
            preds = self._model.predict_entities(chunk, labels, threshold=threshold)
            results.append(list(preds) if isinstance(preds, list) else [])
        return results

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
