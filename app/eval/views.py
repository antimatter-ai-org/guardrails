from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.eval.cache_paths import EvalCachePaths
from app.eval.suite_loader import DatasetSpec


@dataclass(frozen=True, slots=True)
class WhereClause:
    field: str
    op: str  # "==" | "!=" | ">" | ">=" | "<" | "<="
    value: str


@dataclass(frozen=True, slots=True)
class ViewSpec:
    base_split: str
    where: tuple[WhereClause, ...]
    max_samples: int | None
    seed: int
    stratify_by: tuple[str, ...]
    view_name: str | None = None

    def cache_key(self) -> str:
        payload = {
            "base_split": self.base_split,
            "where": [{"field": x.field, "op": x.op, "value": x.value} for x in self.where],
            "max_samples": self.max_samples,
            "seed": self.seed,
            "stratify_by": list(self.stratify_by),
            "view_name": self.view_name,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        return digest


def _dataset_slug(dataset_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", dataset_id).strip("_").lower()


def _parse_where(expr: str) -> WhereClause:
    expr = str(expr).strip()
    for op in ("==", "!=", ">=", "<=", ">", "<"):
        if op in expr:
            left, right = expr.split(op, 1)
            field = left.strip()
            value = right.strip()
            if not field:
                raise ValueError(f"invalid --where (missing field): {expr}")
            if value == "":
                raise ValueError(f"invalid --where (missing value): {expr}")
            return WhereClause(field=field, op=op, value=value)
    # Convenience: field=value means equality.
    if "=" in expr:
        left, right = expr.split("=", 1)
        field = left.strip()
        value = right.strip()
        if not field or value == "":
            raise ValueError(f"invalid --where: {expr}")
        return WhereClause(field=field, op="==", value=value)
    raise ValueError(f"invalid --where: {expr}")


def parse_where_clauses(items: list[str] | None) -> tuple[WhereClause, ...]:
    if not items:
        return tuple()
    return tuple(_parse_where(x) for x in items)


def _coerce_literal(value: str) -> Any:
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v


def _compare(a: Any, op: str, b: Any) -> bool:
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    try:
        if op == ">":
            return a > b
        if op == ">=":
            return a >= b
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
    except Exception:
        return False
    return False


def _row_value(
    *,
    row: dict[str, Any],
    field: str,
    scored_entity_count: int | None,
) -> Any:
    if field == "__scored_entity_count__":
        return scored_entity_count
    return row.get(field)


def _group_key(
    *,
    row: dict[str, Any],
    fields: tuple[str, ...],
    scored_entity_count: int | None,
    label_presence_key: str | None,
) -> tuple[Any, ...]:
    key: list[Any] = []
    for f in fields:
        if f == "label_presence":
            key.append(label_presence_key or "")
            continue
        key.append(_row_value(row=row, field=f, scored_entity_count=scored_entity_count))
    return tuple(key)


def _stratified_sample(
    *,
    groups: dict[tuple[Any, ...], list[int]],
    max_samples: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    total = sum(len(v) for v in groups.values())
    if total <= 0 or max_samples <= 0:
        return []
    if total <= max_samples:
        out = [idx for items in groups.values() for idx in items]
        return sorted(out)

    # Shuffle in-group deterministically.
    for items in groups.values():
        rng.shuffle(items)

    # Proportional allocation with at-least-1 for non-empty groups.
    keys = list(groups.keys())
    targets: dict[tuple[Any, ...], int] = {}
    remaining = max_samples
    non_empty = [k for k in keys if groups[k]]
    for k in non_empty:
        targets[k] = 1
        remaining -= 1
    if remaining < 0:
        # Too many groups; fall back to selecting one per group until exhausted.
        chosen: list[int] = []
        for k in non_empty[:max_samples]:
            chosen.extend(groups[k][:1])
        return sorted(chosen)

    # Allocate rest proportionally.
    for k in non_empty:
        if remaining <= 0:
            break
        frac = len(groups[k]) / total
        add = int(round(frac * remaining))
        targets[k] += max(0, add)

    # Clamp to group sizes and fix total.
    chosen: list[int] = []
    for k in non_empty:
        take = min(len(groups[k]), targets.get(k, 0))
        chosen.extend(groups[k][:take])

    if len(chosen) > max_samples:
        rng.shuffle(chosen)
        chosen = chosen[:max_samples]
    elif len(chosen) < max_samples:
        # Fill from remaining pool.
        remaining_pool: list[int] = []
        chosen_set = set(chosen)
        for k in non_empty:
            remaining_pool.extend([idx for idx in groups[k] if idx not in chosen_set])
        rng.shuffle(remaining_pool)
        chosen.extend(remaining_pool[: max_samples - len(chosen)])

    return sorted(set(chosen))


def resolve_view_indices(
    *,
    dataset_rows: Any,  # datasets.Dataset
    dataset_id: str,
    dataset_fingerprint: str,
    spec: DatasetSpec,
    cache_paths: EvalCachePaths,
    view: ViewSpec,
    scored_entity_count_fn: Callable[[dict[str, Any]], int],
    label_presence_fn: Callable[[dict[str, Any]], str] | None,
) -> tuple[list[int] | None, dict[str, Any]]:
    """
    Returns:
      - indices: None means "use all rows"
      - meta: view metadata for reporting
    """
    if not view.where and view.max_samples is None and not view.stratify_by and not view.view_name:
        return None, {"type": "base_split", "base_split": view.base_split}

    view_dir = cache_paths.views / _dataset_slug(dataset_id)
    view_dir.mkdir(parents=True, exist_ok=True)
    cache_key = view.cache_key()
    prefix = (view.view_name or "view").strip().lower()
    cache_file = view_dir / f"{prefix}_{cache_key}.json"

    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if str(payload.get("dataset_fingerprint") or "") == str(dataset_fingerprint or ""):
            indices = [int(x) for x in (payload.get("indices") or [])]
            return indices, {
                "type": "cached_view",
                "cache_path": str(cache_file),
                "from_cache": True,
                "base_split": view.base_split,
                "where": [{"field": x.field, "op": x.op, "value": x.value} for x in view.where],
                "max_samples": view.max_samples,
                "seed": view.seed,
                "stratify_by": list(view.stratify_by),
                "view_name": view.view_name,
            }

    # Compute indices by scanning rows once.
    candidate_indices: list[int] = []
    groups: dict[tuple[Any, ...], list[int]] = {}

    for idx, row in enumerate(dataset_rows):
        if not isinstance(row, dict):
            try:
                row = dict(row)
            except Exception:
                continue
        scored_count = scored_entity_count_fn(row)
        label_presence_key = label_presence_fn(row) if label_presence_fn else None

        ok = True
        for clause in view.where:
            left = _row_value(row=row, field=clause.field, scored_entity_count=scored_count)
            right = _coerce_literal(clause.value)
            if not _compare(left, clause.op, right):
                ok = False
                break
        if not ok:
            continue

        if view.stratify_by:
            key = _group_key(
                row=row,
                fields=view.stratify_by,
                scored_entity_count=scored_count,
                label_presence_key=label_presence_key,
            )
            groups.setdefault(key, []).append(idx)
        else:
            candidate_indices.append(idx)

    if view.stratify_by:
        selected = _stratified_sample(
            groups=groups,
            max_samples=view.max_samples or sum(len(v) for v in groups.values()),
            seed=view.seed,
        )
    else:
        selected = candidate_indices
        if view.max_samples is not None and len(selected) > view.max_samples:
            rng = random.Random(view.seed)
            rng.shuffle(selected)
            selected = sorted(selected[: view.max_samples])

    payload = {
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "base_split": view.base_split,
        "where": [{"field": x.field, "op": x.op, "value": x.value} for x in view.where],
        "max_samples": view.max_samples,
        "seed": view.seed,
        "stratify_by": list(view.stratify_by),
        "view_name": view.view_name,
        "indices": selected,
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return selected, {
        "type": "cached_view",
        "cache_path": str(cache_file),
        "from_cache": False,
        "base_split": view.base_split,
        "where": [{"field": x.field, "op": x.op, "value": x.value} for x in view.where],
        "max_samples": view.max_samples,
        "seed": view.seed,
        "stratify_by": list(view.stratify_by),
        "view_name": view.view_name,
    }


def builtin_view(name: str) -> tuple[list[WhereClause], str]:
    n = str(name or "").strip().lower()
    if n == "negative":
        return ([_parse_where("__scored_entity_count__==0")], "negative")
    raise ValueError(f"unknown view: {name}")
