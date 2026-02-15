from __future__ import annotations

import logging
import re
from typing import Any

from app.eval.datasets.base import DatasetAdapter
from app.eval.datasets.synthetic_split import load_or_create_synthetic_split
from app.eval.labels import canonicalize_rubai_gold_label
from app.eval.types import EvalSample, EvalSpan

logger = logging.getLogger(__name__)

_OUTSIDE_LABELS = {"TEXT", "O", "", "NONE"}
_MAX_ALIGNMENT_WARNINGS = 20
_alignment_warning_count = 0


def _token_offsets(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _warn_alignment(message: str, *args: object) -> None:
    global _alignment_warning_count
    if _alignment_warning_count >= _MAX_ALIGNMENT_WARNINGS:
        return
    _alignment_warning_count += 1
    logger.warning(message, *args)
    if _alignment_warning_count == _MAX_ALIGNMENT_WARNINGS:
        logger.warning("rubai alignment warnings suppressed after %d rows", _MAX_ALIGNMENT_WARNINGS)


def _build_spans_from_token_types(text: str, token_types: list[str]) -> list[EvalSpan]:
    offsets = _token_offsets(text)
    if not offsets or not token_types:
        return []

    max_len = min(len(offsets), len(token_types))

    spans: list[EvalSpan] = []
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    def flush() -> None:
        nonlocal current_label, current_start, current_end
        if current_label is None or current_start is None or current_end is None:
            return
        spans.append(
            EvalSpan(
                start=current_start,
                end=current_end,
                label=current_label,
                canonical_label=canonicalize_rubai_gold_label(current_label),
            )
        )
        current_label = None
        current_start = None
        current_end = None

    for idx in range(max_len):
        raw_label = str(token_types[idx]).strip()
        label = raw_label.upper()
        token_start, token_end = offsets[idx]

        if label in _OUTSIDE_LABELS:
            flush()
            continue

        if current_label == label and current_end is not None:
            current_end = token_end
            continue

        flush()
        current_label = label
        current_start = token_start
        current_end = token_end

    flush()
    return spans


def _select_token_types(
    text: str,
    row_types: list[str],
    denorm_types: list[str],
    labels: list[str],
) -> list[str]:
    token_count = len(_token_offsets(text))
    if token_count == 0:
        return []
    if len(row_types) == token_count:
        return row_types
    if len(denorm_types) == token_count:
        return denorm_types

    normalized_labels = {str(item).strip().upper() for item in labels}
    if len(row_types) == 1 and row_types[0].strip().upper() in _OUTSIDE_LABELS and normalized_labels <= _OUTSIDE_LABELS:
        return ["TEXT"] * token_count

    if row_types and abs(len(row_types) - token_count) <= 2:
        _warn_alignment("rubai token alignment adjusted (tokens=%d, types=%d)", token_count, len(row_types))
        if len(row_types) < token_count:
            return row_types + ["TEXT"] * (token_count - len(row_types))
        return row_types[:token_count]

    _warn_alignment(
        "rubai token alignment mismatch dropped (tokens=%d, types=%d, denorm=%d, labels=%s)",
        token_count,
        len(row_types),
        len(denorm_types),
        sorted(normalized_labels),
    )
    return ["TEXT"] * token_count


class RubaiNerPersonalAdapter(DatasetAdapter):
    @property
    def supports_synthetic_split(self) -> bool:
        return True

    @property
    def dataset_name(self) -> str:
        return "BoburAmirov/rubai-NER-150K-Personal"

    @staticmethod
    def _row_label_set(row: dict[str, Any]) -> set[str]:
        canonical_labels: set[str] = set()
        labels = row.get("labels") or []
        for item in labels:
            raw = str(item).strip().upper()
            if raw in _OUTSIDE_LABELS:
                continue
            canonical = canonicalize_rubai_gold_label(raw)
            if canonical:
                canonical_labels.add(str(canonical))
        return canonical_labels

    @staticmethod
    def _rows_to_samples(
        rows: Any,
        split_name: str,
        max_samples: int | None = None,
    ) -> list[EvalSample]:
        samples: list[EvalSample] = []
        for idx, row in enumerate(rows):
            sample_id = str(row.get("id", idx))
            text = str(row.get("original", ""))
            row_types = [str(item) for item in (row.get("types") or [])]
            denorm_types = [str(item) for item in (row.get("denorm_types") or [])]
            labels = [str(item) for item in (row.get("labels") or [])]
            token_types = _select_token_types(text, row_types, denorm_types, labels)
            spans = _build_spans_from_token_types(text, token_types)

            samples.append(
                EvalSample(
                    sample_id=sample_id,
                    text=text,
                    gold_spans=spans,
                    metadata={
                        "source": row.get("domain"),
                        "noisy": None,
                        "__split__": split_name,
                    },
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
        return samples

    def load_samples(
        self,
        split: str,
        cache_dir: str,
        hf_token: str | None,
        synthetic_test_size: float = 0.2,
        synthetic_split_seed: int = 42,
        max_samples: int | None = None,
    ) -> list[EvalSample]:
        try:
            from datasets import get_dataset_split_names, load_dataset
        except Exception as exc:
            raise RuntimeError("datasets package is required. Install with guardrails-service[eval].") from exc

        available_splits = set(get_dataset_split_names(self.dataset_name, token=hf_token))
        if split in available_splits:
            dataset = load_dataset(
                self.dataset_name,
                split=split,
                token=hf_token,
                cache_dir=cache_dir,
            )
            return self._rows_to_samples(dataset, split_name=split, max_samples=max_samples)

        if split in {"train", "test"} and "train" in available_splits:
            full_train = load_dataset(
                self.dataset_name,
                split="train",
                token=hf_token,
                cache_dir=cache_dir,
            )
            dataset_fingerprint = str(getattr(full_train, "_fingerprint", ""))
            sample_count = len(full_train)
            try:
                split_result = load_or_create_synthetic_split(
                    dataset_name=self.dataset_name,
                    cache_dir=cache_dir,
                    sample_count=sample_count,
                    test_size=synthetic_test_size,
                    seed=synthetic_split_seed,
                    dataset_fingerprint=dataset_fingerprint,
                )
            except ValueError:
                label_sets = [self._row_label_set(row) for row in full_train]
                split_result = load_or_create_synthetic_split(
                    dataset_name=self.dataset_name,
                    cache_dir=cache_dir,
                    sample_count=sample_count,
                    sample_labels=label_sets,
                    test_size=synthetic_test_size,
                    seed=synthetic_split_seed,
                    dataset_fingerprint=dataset_fingerprint,
                )
            indices = split_result.test_indices if split == "test" else split_result.train_indices
            if max_samples is not None:
                indices = indices[:max_samples]

            logger.info(
                "dataset %s uses synthetic %s split from train (cached=%s, cache=%s)",
                self.dataset_name,
                split,
                split_result.from_cache,
                split_result.cache_path,
            )
            subset = full_train.select(indices)
            return self._rows_to_samples(subset, split_name=split, max_samples=None)

        if available_splits:
            fallback_split = sorted(available_splits)[0]
            logger.warning(
                "dataset %s has no split '%s', using '%s'",
                self.dataset_name,
                split,
                fallback_split,
            )
            dataset = load_dataset(
                self.dataset_name,
                split=fallback_split,
                token=hf_token,
                cache_dir=cache_dir,
            )
            return self._rows_to_samples(dataset, split_name=fallback_split, max_samples=max_samples)

        raise RuntimeError(f"No splits found for dataset {self.dataset_name}")
