from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    dataset_id: str
    hf_id: str
    kind: str
    text_field: str
    spans_field: str
    scored_labels: frozenset[str]
    label_map: dict[str, str]
    slice_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SuiteConfig:
    name: str
    title: str
    default_split: str
    datasets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalRegistry:
    version: int
    suites: dict[str, SuiteConfig]
    datasets: dict[str, DatasetConfig]


def default_registry_path() -> Path:
    return Path("configs") / "eval" / "suites.yaml"


def _as_str_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return [str(item) for item in value]


def load_eval_registry(path: str | Path | None = None) -> EvalRegistry:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required for eval. Install project dependencies.") from exc

    cfg_path = Path(path) if path is not None else default_registry_path()
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("eval registry root must be a mapping")

    version = int(payload.get("version", 0) or 0)
    if version != 1:
        raise ValueError(f"unsupported eval registry version: {version}")

    suites_raw = payload.get("suites") or {}
    datasets_raw = payload.get("datasets") or {}
    if not isinstance(suites_raw, dict) or not isinstance(datasets_raw, dict):
        raise ValueError("eval registry requires 'suites' and 'datasets' mappings")

    suites: dict[str, SuiteConfig] = {}
    for suite_name, suite_payload in suites_raw.items():
        if not isinstance(suite_payload, dict):
            raise ValueError(f"suite '{suite_name}' must be a mapping")
        suites[str(suite_name)] = SuiteConfig(
            name=str(suite_name),
            title=str(suite_payload.get("title") or suite_name),
            default_split=str(suite_payload.get("default_split") or "fast"),
            datasets=tuple(_as_str_list(suite_payload.get("datasets"), field=f"suites.{suite_name}.datasets")),
        )

    datasets: dict[str, DatasetConfig] = {}
    for dataset_id, ds_payload in datasets_raw.items():
        if not isinstance(ds_payload, dict):
            raise ValueError(f"dataset '{dataset_id}' must be a mapping")

        scored_labels = frozenset(_as_str_list(ds_payload.get("scored_labels"), field=f"datasets.{dataset_id}.scored_labels"))
        label_map_raw = ds_payload.get("label_map") or {}
        if not isinstance(label_map_raw, dict):
            raise ValueError(f"datasets.{dataset_id}.label_map must be a mapping")
        label_map: dict[str, str] = {str(k): str(v) for k, v in label_map_raw.items()}

        slice_fields = tuple(_as_str_list(ds_payload.get("slice_fields"), field=f"datasets.{dataset_id}.slice_fields"))

        datasets[str(dataset_id)] = DatasetConfig(
            dataset_id=str(dataset_id),
            hf_id=str(ds_payload.get("hf_id") or dataset_id),
            kind=str(ds_payload.get("kind") or "hf_span_v1"),
            text_field=str(ds_payload.get("text_field") or "source_text"),
            spans_field=str(ds_payload.get("spans_field") or "privacy_mask"),
            scored_labels=scored_labels,
            label_map=label_map,
            slice_fields=slice_fields,
        )

    # Validate suite references.
    for suite in suites.values():
        missing = [ds_id for ds_id in suite.datasets if ds_id not in datasets]
        if missing:
            raise ValueError(f"suite '{suite.name}' references unknown datasets: {missing}")

    return EvalRegistry(version=version, suites=suites, datasets=datasets)

