from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.eval.types import EvalSample, EvalSpan
from app.eval_v3.taxonomy import normalize_label


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    dataset_id: str
    split: str
    samples: list[EvalSample]
    dataset_fingerprint: str | None
    selected_indices: list[int]
    indices_cache_path: str | None
    indices_from_cache: bool


def _dataset_slug(name: str) -> str:
    safe = []
    for char in name.lower():
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def _map_gold_label(raw: Any, label_map: dict[str, str]) -> str | None:
    raw_str = str(raw).strip()
    if not raw_str:
        return None

    # Explicit mapping has priority (try exact + uppercase).
    direct = label_map.get(raw_str)
    if direct is None:
        direct = label_map.get(raw_str.upper())
    if direct is not None:
        return normalize_label(direct)

    # If it's already canonical, accept.
    return normalize_label(raw_str)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _extract_spans(row: dict[str, Any], spans_field: str, label_map: dict[str, str]) -> list[EvalSpan]:
    raw_spans = row.get(spans_field)
    if raw_spans is None:
        return []
    if not isinstance(raw_spans, list):
        raise ValueError(f"expected '{spans_field}' to be a list")

    spans: list[EvalSpan] = []
    for item in raw_spans:
        if not isinstance(item, dict):
            continue
        start = _as_int(item.get("start"))
        end = _as_int(item.get("end"))
        raw_label = item.get("label")
        if start is None or end is None or end <= start:
            continue
        canonical = _map_gold_label(raw_label, label_map)
        spans.append(
            EvalSpan(
                start=start,
                end=end,
                label=str(raw_label) if raw_label is not None else "UNKNOWN",
                canonical_label=canonical,
                score=None,
                detector=None,
            )
        )
    return spans


def load_hf_split(
    *,
    hf_id: str,
    split: str,
    cache_dir: str,
    hf_token: str | bool | None,
) -> tuple[Any, str | None]:
    try:
        from datasets import load_dataset, load_from_disk  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets package is required. Install with guardrails-service[eval].") from exc

    # Support local datasets saved via `DatasetDict.save_to_disk`.
    path = Path(str(hf_id)).expanduser()
    if path.exists():
        marker_files = ("dataset_dict.json", "state.json", "dataset_info.json")
        if any((path / name).exists() for name in marker_files):
            ds = load_from_disk(str(path))
            # Handle DatasetDict or Dataset
            if hasattr(ds, "keys") and split in getattr(ds, "keys")():
                ds = ds[split]
            fingerprint = str(getattr(ds, "_fingerprint", "")) or None
            return ds, fingerprint

    ds = load_dataset(
        hf_id,
        split=split,
        token=hf_token,
        cache_dir=cache_dir,
    )
    fingerprint = str(getattr(ds, "_fingerprint", "")) or None
    return ds, fingerprint


def scan_hf_span_dataset(
    *,
    ds: Any,
    spans_field: str,
    label_map: dict[str, str],
) -> tuple[list[int | None], list[str | None], list[str | None], list[set[str]]]:
    entity_counts: list[int | None] = []
    languages: list[str | None] = []
    script_profiles: list[str | None] = []
    label_sets: list[set[str]] = []

    for row in ds:
        if not isinstance(row, dict):
            entity_counts.append(None)
            languages.append(None)
            script_profiles.append(None)
            label_sets.append(set())
            continue

        entity_counts.append(_as_int(row.get("entity_count")))

        lang = row.get("language")
        languages.append(str(lang) if isinstance(lang, str) else None)

        sp = row.get("script_profile")
        script_profiles.append(str(sp) if isinstance(sp, str) else None)

        present: set[str] = set()
        raw_spans = row.get(spans_field) or []
        if isinstance(raw_spans, list):
            for item in raw_spans:
                if not isinstance(item, dict):
                    continue
                canonical = _map_gold_label(item.get("label"), label_map)
                if canonical:
                    present.add(canonical)
        label_sets.append(present)

    return entity_counts, languages, script_profiles, label_sets


def build_samples_from_hf_split(
    *,
    dataset_id: str,
    split: str,
    ds: Any,
    text_field: str,
    spans_field: str,
    label_map: dict[str, str],
    slice_fields: tuple[str, ...],
    selected_indices: list[int] | None,
    max_samples: int | None,
) -> list[EvalSample]:
    if selected_indices is not None:
        if max_samples is not None:
            selected_indices = selected_indices[: max_samples]
        ds = ds.select(list(selected_indices))

    samples: list[EvalSample] = []
    for local_idx, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        text = str(row.get(text_field, ""))

        uid = row.get("source_uid")
        if uid is None:
            uid = row.get("source_row_idx")
        if uid is None:
            uid = row.get("id")
        if uid is None:
            uid = local_idx
        sample_id = f"{_dataset_slug(dataset_id)}::{split}::{uid}"

        spans = _extract_spans(row, spans_field=spans_field, label_map=label_map)
        metadata: dict[str, str | bool | int | float | None] = {"__dataset__": dataset_id, "__split__": split}
        for field in slice_fields:
            if field in {text_field, spans_field}:
                continue
            val = row.get(field)
            if isinstance(val, (str, int, float)) or val is None:
                metadata[field] = val
            elif isinstance(val, bool):
                metadata[field] = _as_bool(val)
            else:
                metadata[field] = str(val)

        for common in ("language", "script_profile", "entity_count", "source", "noisy"):
            if common in metadata:
                continue
            if common not in row:
                continue
            val = row.get(common)
            if isinstance(val, (str, int, float, bool)) or val is None:
                metadata[common] = val

        samples.append(EvalSample(sample_id=sample_id, text=text, gold_spans=spans, metadata=metadata))
        if max_samples is not None and selected_indices is None and len(samples) >= max_samples:
            break
    return samples


def load_hf_span_dataset(
    *,
    dataset_id: str,
    hf_id: str,
    split: str,
    cache_dir: str,
    hf_token: str | bool | None,
    text_field: str,
    spans_field: str,
    label_map: dict[str, str],
    slice_fields: tuple[str, ...],
    selected_indices: list[int] | None = None,
    max_samples: int | None = None,
    indices_cache_path: str | None = None,
    indices_from_cache: bool = False,
) -> LoadedDataset:
    ds, fingerprint = load_hf_split(hf_id=hf_id, split=split, cache_dir=cache_dir, hf_token=hf_token)
    samples = build_samples_from_hf_split(
        dataset_id=dataset_id,
        split=split,
        ds=ds,
        text_field=text_field,
        spans_field=spans_field,
        label_map=label_map,
        slice_fields=slice_fields,
        selected_indices=selected_indices,
        max_samples=max_samples,
    )

    return LoadedDataset(
        dataset_id=dataset_id,
        split=split,
        samples=samples,
        dataset_fingerprint=fingerprint,
        selected_indices=selected_indices or list(range(len(samples))),
        indices_cache_path=indices_cache_path,
        indices_from_cache=indices_from_cache,
    )
