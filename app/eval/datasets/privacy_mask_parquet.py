from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.core.analysis.mapping import canonicalize_entity_type
from app.core.labels import CANONICAL_PII_LABELS
from app.eval.cache_paths import EvalCachePaths
from app.eval.suite_loader import DatasetSpec
from app.eval.types import EvalSample, EvalSpan


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    dataset_id: str
    split: str
    available_splits: tuple[str, ...]
    fingerprint: str
    rows: Any  # datasets.Dataset


def _stable_sample_id(*, dataset_id: str, split: str, source_row_idx: int | str | None, fallback_idx: int) -> str:
    if source_row_idx is not None:
        raw = f"{dataset_id}::{split}::{source_row_idx}"
    else:
        raw = f"{dataset_id}::{split}::idx::{fallback_idx}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"s_{digest}"


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def canonicalize_gold_label(*, raw_label: str, mapping: dict[str, str]) -> str | None:
    key = str(raw_label or "").strip().lower()
    if not key:
        return None
    mapped = mapping.get(key)
    if mapped:
        return str(mapped).strip().lower()
    if key in CANONICAL_PII_LABELS:
        return key
    # Fall back to runtime-style canonicalization for common placeholder labels.
    return canonicalize_entity_type(key)


def _configure_hf_env(cache_paths: EvalCachePaths) -> None:
    # Keep these aligned with datasets/huggingface_hub defaults to reuse caches across runs.
    import os

    os.environ.setdefault("HF_HOME", str(cache_paths.hf))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_paths.hf / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_paths.hf / "datasets"))


def load_hf_dataset(
    *,
    dataset_id: str,
    split: str,
    cache_paths: EvalCachePaths,
    hf_token: str | None,
) -> LoadedDataset:
    _configure_hf_env(cache_paths)
    try:
        from datasets import get_dataset_split_names, load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets package is required. Install with guardrails-service[eval].") from exc

    try:
        split_names = [str(x) for x in get_dataset_split_names(dataset_id, token=hf_token)]
    except Exception:
        split_names = []

    rows = load_dataset(
        dataset_id,
        split=split,
        token=hf_token,
        cache_dir=str(cache_paths.hf / "datasets"),
    )
    fingerprint = str(getattr(rows, "_fingerprint", "")) or ""
    return LoadedDataset(
        dataset_id=dataset_id,
        split=split,
        available_splits=tuple(sorted(set(split_names))),
        fingerprint=fingerprint,
        rows=rows,
    )


def rows_to_samples(
    *,
    dataset: LoadedDataset,
    spec: DatasetSpec,
    scored_labels: set[str],
    indices: Iterable[int] | None,
) -> list[EvalSample]:
    rows = dataset.rows
    if indices is not None:
        idx_list = list(indices)
        selector = getattr(rows, "select", None)
        if callable(selector):
            rows = selector(idx_list)
        else:
            # Primarily for tests; production uses datasets.Dataset which supports .select().
            rows = [rows[i] for i in idx_list]

    annotated = set(spec.annotated_labels)
    mapping = spec.gold_label_mapping

    samples: list[EvalSample] = []
    for idx, row in enumerate(rows):
        text = str(row.get(spec.text_field, "") or "")
        mask_items = row.get(spec.mask_field) or []

        spans: list[EvalSpan] = []
        for item in mask_items:
            if not isinstance(item, dict):
                continue
            start = _as_int(item.get("start"))
            end = _as_int(item.get("end"))
            raw_label = str(item.get("label") or "").strip()
            if start is None or end is None or end <= start:
                continue
            canonical = canonicalize_gold_label(raw_label=raw_label, mapping=mapping)
            if canonical is None:
                continue
            if canonical not in annotated:
                # Dataset spec says it doesn't annotate this label (even if present in raw).
                continue
            if canonical not in scored_labels:
                continue
            # Clamp to avoid exceptions on bad offsets; keep conservative behavior.
            start = max(0, min(start, len(text)))
            end = max(0, min(end, len(text)))
            if end <= start:
                continue
            spans.append(EvalSpan(start=start, end=end, label=raw_label, canonical_label=canonical))

        source_row_idx = row.get("source_row_idx", row.get("id", None))
        sample_id = _stable_sample_id(
            dataset_id=dataset.dataset_id,
            split=dataset.split,
            source_row_idx=source_row_idx,
            fallback_idx=idx,
        )

        metadata: dict[str, str | bool | int | float | None] = {
            "__dataset_id__": dataset.dataset_id,
            "__split__": dataset.split,
        }
        for field in spec.slice_fields:
            if field in row:
                value = row.get(field)
                if isinstance(value, (str, bool, int, float)) or value is None:
                    metadata[field] = value
                else:
                    # Keep metadata JSON-friendly and compact.
                    metadata[field] = json.dumps(value, ensure_ascii=False)[:512]
        metadata["__scored_gold_spans__"] = int(len(spans))

        samples.append(
            EvalSample(
                sample_id=sample_id,
                text=text,
                gold_spans=spans,
                metadata=metadata,
            )
        )

    return samples
