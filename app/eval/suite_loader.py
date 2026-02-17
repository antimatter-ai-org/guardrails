from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.labels import CANONICAL_PII_LABELS


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_id: str
    format: str
    text_field: str
    mask_field: str
    annotated_labels: tuple[str, ...]
    gold_label_mapping: dict[str, str]
    slice_fields: tuple[str, ...]
    tags: tuple[str, ...]
    notes: str

    def annotated_label_set(self) -> set[str]:
        return set(self.annotated_labels)


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    suite_id: str
    default_collection: str
    default_split: str
    scored_labels: tuple[str, ...]
    datasets: tuple[DatasetSpec, ...]

    def dataset_by_id(self) -> dict[str, DatasetSpec]:
        return {item.dataset_id: item for item in self.datasets}


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(f"expected list, got {type(value).__name__}")


def load_suite(path: str | Path) -> SuiteSpec:
    suite_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid suite yaml: {suite_path}")

    suite_id = str(payload.get("suite_id") or "").strip()
    if not suite_id:
        raise ValueError(f"suite_id is required: {suite_path}")

    default_collection = str(payload.get("default_collection") or "").strip()
    default_split = str(payload.get("default_split") or "fast").strip()

    scored_labels_raw = _as_str_list(payload.get("scored_labels")) or list(CANONICAL_PII_LABELS)
    scored_labels = tuple(str(item).strip().lower() for item in scored_labels_raw if str(item).strip())

    datasets_raw = payload.get("datasets") or []
    if not isinstance(datasets_raw, list):
        raise ValueError(f"datasets must be a list: {suite_path}")

    datasets: list[DatasetSpec] = []
    for item in datasets_raw:
        if not isinstance(item, dict):
            raise ValueError(f"dataset item must be a dict: {suite_path}")
        dataset_id = str(item.get("id") or "").strip()
        if not dataset_id:
            raise ValueError(f"dataset id is required: {suite_path}")
        fmt = str(item.get("format") or "").strip()
        if not fmt:
            raise ValueError(f"dataset format is required: {dataset_id}")
        text_field = str(item.get("text_field") or "source_text").strip()
        mask_field = str(item.get("mask_field") or "privacy_mask").strip()
        annotated_labels = tuple(str(x).strip().lower() for x in _as_str_list(item.get("annotated_labels")))

        mapping_raw = item.get("gold_label_mapping") or {}
        if not isinstance(mapping_raw, dict):
            raise ValueError(f"gold_label_mapping must be a dict: {dataset_id}")
        gold_label_mapping = {str(k).strip().lower(): str(v).strip().lower() for k, v in mapping_raw.items()}

        slice_fields = tuple(str(x).strip() for x in _as_str_list(item.get("slice_fields")))
        tags = tuple(str(x).strip().lower() for x in _as_str_list(item.get("tags")))
        notes = str(item.get("notes") or "").strip()

        datasets.append(
            DatasetSpec(
                dataset_id=dataset_id,
                format=fmt,
                text_field=text_field,
                mask_field=mask_field,
                annotated_labels=annotated_labels,
                gold_label_mapping=gold_label_mapping,
                slice_fields=slice_fields,
                tags=tags,
                notes=notes,
            )
        )

    return SuiteSpec(
        suite_id=suite_id,
        default_collection=default_collection,
        default_split=default_split,
        scored_labels=scored_labels,
        datasets=tuple(datasets),
    )

