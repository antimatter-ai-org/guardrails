from __future__ import annotations

import json
import re
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from app.runtime.gliner_word_chunking import (
    build_prompt_tokens_for_length_check,
    chunk_text_by_gliner_words,
    deterministic_overlap_words,
    gliner_prompt_len_words,
)
from app.runtime.tokenizer_chunking import TextChunkWindow
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
    def ensure_ready(self, timeout_s: float | None) -> bool:
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
        self._encoder_max_len: int | None = None
        self._gliner_cfg: dict[str, Any] | None = None
        self._load_error: str | None = None
        self._load_model()

    @staticmethod
    def _read_gliner_config(model_source: str) -> dict[str, Any] | None:
        try:
            base = Path(str(model_source))
        except Exception:
            return None
        if not base.exists() or not base.is_dir():
            # Try to fetch config from the Hub when a local dir is not provided.
            try:
                from huggingface_hub import hf_hub_download  # type: ignore
            except Exception:
                return None
            try:
                cfg_path = hf_hub_download(repo_id=str(model_source), filename="gliner_config.json")
                return json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            except Exception:
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
        if hasattr(cfg, "to_dict"):
            try:
                return dict(cfg.to_dict())
            except Exception:
                return None
        if hasattr(cfg, "__dict__"):
            try:
                return dict(getattr(cfg, "__dict__"))
            except Exception:
                return None
        return None

    @staticmethod
    def _reasonable_max_length(value: Any) -> int | None:
        try:
            as_int = int(value)
        except Exception:
            return None
        if as_int < 2:
            return None
        # Many tokenizers use huge sentinels for "infinite".
        if as_int > 100_000:
            return None
        return as_int

    @classmethod
    def _resolve_encoder_max_len(cls, *, encoder_name: str, encoder_tokenizer: Any) -> int:
        # GLiNER internally sets truncation=True without explicit max_length.
        # Refuse to risk silent truncation by ensuring prompt+chunk stay within the
        # encoder's real context length.
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
            # GLiNER warning text (gliner/data_processing/processor.py) includes:
            # "Sentence of length X has been truncated to Y"
            if re.search(r"Sentence of length .*truncated to", text, flags=re.IGNORECASE):
                return text
        return None

    def _prompt_tokens(self, *, labels: list[str]) -> list[str]:
        cfg = self._gliner_cfg or {}
        ent_token = str(cfg.get("ent_token") or cfg.get("entity_token") or "<ENT>")
        sep_token = str(cfg.get("sep_token") or cfg.get("separator_token") or "<SEP>")
        return build_prompt_tokens_for_length_check(ent_token=ent_token, sep_token=sep_token, labels=labels)

    def _prompt_str_for_length_check(self, *, labels: list[str]) -> str:
        return " ".join(self._prompt_tokens(labels=labels))

    def _would_tokenizer_truncate(self, *, prompt_str: str, chunk_text: str) -> bool:
        if self._encoder_tokenizer is None or self._encoder_max_len is None:
            raise RuntimeError(self._load_error or "gliner runtime is not ready")
        full = prompt_str if not chunk_text else (prompt_str + " " + chunk_text if prompt_str else chunk_text)
        enc = self._encoder_tokenizer(full, add_special_tokens=True, truncation=False)
        input_ids = enc.get("input_ids") if isinstance(enc, dict) else getattr(enc, "input_ids", None)
        if input_ids is None:
            raise RuntimeError("unable to inspect tokenizer output (refuse to risk silent truncation)")
        return len(input_ids) > int(self._encoder_max_len)

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
            # Strict chunking: load GLiNER config + encoder tokenizer for truncation guards.
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
            if not bool(getattr(encoder_tokenizer, "is_fast", False)):
                raise RuntimeError("fast tokenizer is required for safe chunking (refuse to risk silent truncation)")
            encoder_max_len = self._resolve_encoder_max_len(encoder_name=encoder_name, encoder_tokenizer=encoder_tokenizer)
            self._encoder_tokenizer = encoder_tokenizer
            self._encoder_max_len = int(encoder_max_len)
            self._gliner_cfg = dict(gliner_cfg)
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
        if self._model is None or self._encoder_tokenizer is None or self._encoder_max_len is None or self._gliner_cfg is None:
            raise RuntimeError(self._load_error or "gliner runtime is not ready")

        # GLiNER truncates based on its own "word splitter" length (max_len). Chunk accordingly.
        raw_max_len = self._gliner_cfg.get("max_len")
        raw_max_width = self._gliner_cfg.get("max_width", 10)
        try:
            max_len_words = int(raw_max_len)
            max_width = int(raw_max_width)
        except Exception as exc:
            raise RuntimeError("unable to determine GLiNER max_len/max_width (refuse to risk silent truncation)") from exc
        if max_len_words < 2:
            raise RuntimeError("invalid GLiNER max_len (refuse to risk silent truncation)")

        prompt_len = gliner_prompt_len_words(labels)
        max_text_words = max_len_words - int(prompt_len)
        if max_text_words < 1:
            raise RuntimeError("GLiNER prompt consumes entire context window (refuse to risk truncation)")

        prompt_str = self._prompt_str_for_length_check(labels=labels)
        # If prompt alone can overflow the encoder tokenizer window, we cannot scan safely.
        if self._would_tokenizer_truncate(prompt_str=prompt_str, chunk_text=""):
            raise RuntimeError("GLiNER prompt exceeds encoder context length (refuse to risk truncation)")

        attempt_words = int(max_text_words)
        chunk_predictions: list[list[dict[str, Any]]] | None = None
        windows: list[TextChunkWindow] | None = None
        while True:
            overlap_words = deterministic_overlap_words(max_len_words=attempt_words, max_width=max_width)
            windows = chunk_text_by_gliner_words(text=text, max_text_words=attempt_words, overlap_words=overlap_words)
            chunk_texts = [text[w.text_start : w.text_end] for w in windows]

            # Guard against silent transformer tokenizer truncation.
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
                chunk_predictions = self._predict_batch(chunk_texts, labels, threshold)
            trunc_msg = self._extract_truncation_warning(list(caught))
            if trunc_msg is not None:
                next_words = attempt_words // 2
                if next_words < 1:
                    raise RuntimeError(f"GLiNER truncation could not be eliminated: {trunc_msg}")
                attempt_words = next_words
                continue
            break

        if windows is None or chunk_predictions is None:
            raise RuntimeError("GLiNER runtime failed to produce predictions")
        return self._merge_window_predictions(
            text=text,
            windows=list(windows),
            window_predictions=chunk_predictions,
            default_threshold=threshold,
        )

    def ensure_ready(self, timeout_s: float | None) -> bool:
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
        init_timeout_s: float | None,
        infer_timeout_s: float | None,
    ) -> None:
        self._model_name = triton_model_name
        self._hf_model_name = hf_model_name
        self._pytriton_url = pytriton_url
        # init timeout should dominate only if infer timeout is set.
        # NOTE: init timeout is not an inference wall-clock cap; inference_timeout_s must stay None.
        self._init_timeout_s = float(init_timeout_s) if init_timeout_s is not None else 120.0
        if infer_timeout_s is not None:
            self._init_timeout_s = max(self._init_timeout_s, float(infer_timeout_s))
        self._infer_timeout_s = infer_timeout_s
        self._max_batch_size_hint = 32
        self.device = "cuda"
        self._ready = False
        self._load_error: str | None = None
        self._encoder_tokenizer: Any | None = None
        self._encoder_max_len: int | None = None
        self._gliner_cfg: dict[str, Any] | None = None
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
            if not bool(getattr(encoder_tokenizer, "is_fast", False)):
                raise RuntimeError("fast tokenizer is required for safe chunking (refuse to risk silent truncation)")
            encoder_max_len = LocalCpuGlinerRuntime._resolve_encoder_max_len(  # noqa: SLF001
                encoder_name=encoder_name,
                encoder_tokenizer=encoder_tokenizer,
            )
            self._encoder_tokenizer = encoder_tokenizer
            self._encoder_max_len = int(encoder_max_len)
            self._gliner_cfg = dict(gliner_cfg)
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
        if self._encoder_tokenizer is None or self._encoder_max_len is None or self._gliner_cfg is None:
            self._load_tokenizer()
        if self._encoder_tokenizer is None or self._encoder_max_len is None or self._gliner_cfg is None:
            raise RuntimeError(self._load_error or "pytriton gliner runtime is not ready")

        raw_max_len = self._gliner_cfg.get("max_len")
        raw_max_width = self._gliner_cfg.get("max_width", 10)
        try:
            max_len_words = int(raw_max_len)
            max_width = int(raw_max_width)
        except Exception as exc:
            raise RuntimeError("unable to determine GLiNER max_len/max_width (refuse to risk silent truncation)") from exc
        if max_len_words < 2:
            raise RuntimeError("invalid GLiNER max_len (refuse to risk silent truncation)")

        prompt_len = gliner_prompt_len_words(labels)
        max_text_words = max_len_words - int(prompt_len)
        if max_text_words < 1:
            raise RuntimeError("GLiNER prompt consumes entire context window (refuse to risk truncation)")

        # Client-side defense: avoid silent encoder truncation even before hitting the server.
        cfg = self._gliner_cfg or {}
        ent_token = str(cfg.get("ent_token") or cfg.get("entity_token") or "<ENT>")
        sep_token = str(cfg.get("sep_token") or cfg.get("separator_token") or "<SEP>")
        prompt_str = " ".join(build_prompt_tokens_for_length_check(ent_token=ent_token, sep_token=sep_token, labels=labels))
        if LocalCpuGlinerRuntime._would_tokenizer_truncate(self, prompt_str=prompt_str, chunk_text=""):  # noqa: SLF001
            raise RuntimeError("GLiNER prompt exceeds encoder context length (refuse to risk truncation)")

        attempt_words = int(max_text_words)
        while True:
            overlap_words = deterministic_overlap_words(max_len_words=attempt_words, max_width=max_width)
            windows = chunk_text_by_gliner_words(text=text, max_text_words=attempt_words, overlap_words=overlap_words)
            chunk_texts = [text[w.text_start : w.text_end] for w in windows]
            overflow = False
            for chunk in chunk_texts:
                if LocalCpuGlinerRuntime._would_tokenizer_truncate(self, prompt_str=prompt_str, chunk_text=chunk):  # noqa: SLF001
                    overflow = True
                    break
            if overflow:
                next_words = attempt_words // 2
                if next_words < 1:
                    raise RuntimeError("unable to build GLiNER chunks without tokenizer truncation")
                attempt_words = next_words
                continue
            chunk_predictions = self._predict_batch(chunk_texts, labels, threshold)
            break

        return LocalCpuGlinerRuntime._merge_window_predictions(  # noqa: SLF001
            text=text,
            windows=windows,
            window_predictions=chunk_predictions,
            default_threshold=threshold,
        )

    def ensure_ready(self, timeout_s: float | None) -> bool:
        if self._ready:
            return True
        try:
            wait_for_triton_ready(
                pytriton_url=self._pytriton_url,
                contracts=[self._contract],
                timeout_s=timeout_s,
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
    pytriton_init_timeout_s: float | None,
    pytriton_infer_timeout_s: float | None,
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
