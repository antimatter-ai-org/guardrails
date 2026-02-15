from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SyntheticSplitResult:
    train_indices: list[int]
    test_indices: list[int]
    cache_path: Path
    from_cache: bool


def _dataset_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_").lower()


def _normalized_sample_labels(sample_labels: list[set[str]]) -> list[tuple[str, ...]]:
    normalized: list[tuple[str, ...]] = []
    for labels in sample_labels:
        cleaned = sorted({str(item).strip().lower() for item in labels if str(item).strip()})
        if cleaned:
            normalized.append(tuple(cleaned))
        else:
            normalized.append(("__none__",))
    return normalized


def _resolve_test_count(total_samples: int, test_size: float) -> int:
    if total_samples <= 0:
        return 0
    if total_samples == 1:
        return 1
    clamped_size = max(0.01, min(0.99, float(test_size)))
    candidate = int(round(total_samples * clamped_size))
    return min(max(1, candidate), total_samples - 1)


def _build_balanced_test_indices(
    *,
    sample_labels: list[tuple[str, ...]],
    test_count: int,
    seed: int,
) -> list[int]:
    if test_count <= 0:
        return []
    if not sample_labels:
        return []

    rng = random.Random(seed)
    label_to_candidates: dict[str, list[int]] = defaultdict(list)
    for idx, labels in enumerate(sample_labels):
        for label in labels:
            label_to_candidates[label].append(idx)

    for candidates in label_to_candidates.values():
        rng.shuffle(candidates)

    total = len(sample_labels)
    target_count: dict[str, int] = {
        label: int(round((len(candidates) / total) * test_count))
        for label, candidates in label_to_candidates.items()
    }
    selected_count: dict[str, int] = defaultdict(int)
    selected: set[int] = set()
    cursor: dict[str, int] = defaultdict(int)
    active_labels = set(label_to_candidates.keys())

    while len(selected) < test_count and active_labels:
        best_label: str | None = None
        best_deficit = 0
        best_support = 0
        for label in list(active_labels):
            deficit = target_count[label] - selected_count[label]
            if deficit <= 0:
                continue
            support = len(label_to_candidates[label])
            if (
                best_label is None
                or deficit > best_deficit
                or (deficit == best_deficit and support < best_support)
            ):
                best_label = label
                best_deficit = deficit
                best_support = support

        if best_label is None:
            break

        candidates = label_to_candidates[best_label]
        pos = cursor[best_label]
        while pos < len(candidates) and candidates[pos] in selected:
            pos += 1
        cursor[best_label] = pos
        if pos >= len(candidates):
            active_labels.remove(best_label)
            continue

        sample_idx = candidates[pos]
        selected.add(sample_idx)
        for label in sample_labels[sample_idx]:
            selected_count[label] += 1

    if len(selected) < test_count:
        remaining = [idx for idx in range(len(sample_labels)) if idx not in selected]
        rng.shuffle(remaining)
        selected.update(remaining[: test_count - len(selected)])

    return sorted(selected)


def _cache_file_path(
    *,
    cache_dir: str,
    dataset_name: str,
    sample_count: int,
    test_size: float,
    seed: int,
    dataset_fingerprint: str | None,
) -> Path:
    signature = {
        "dataset": dataset_name,
        "sample_count": int(sample_count),
        "test_size": float(test_size),
        "seed": int(seed),
        "fingerprint": dataset_fingerprint or "",
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    split_dir = Path(cache_dir).expanduser().resolve() / "_synthetic_splits" / _dataset_slug(dataset_name)
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir / f"{key}.json"


def load_or_create_synthetic_split(
    *,
    dataset_name: str,
    cache_dir: str,
    sample_count: int,
    test_size: float,
    seed: int,
    dataset_fingerprint: str | None = None,
    sample_labels: list[set[str]] | None = None,
) -> SyntheticSplitResult:
    test_count = _resolve_test_count(sample_count, test_size)
    cache_path = _cache_file_path(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        sample_count=sample_count,
        test_size=test_size,
        seed=seed,
        dataset_fingerprint=dataset_fingerprint,
    )

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        train_indices = [int(item) for item in payload.get("train_indices", [])]
        test_indices = [int(item) for item in payload.get("test_indices", [])]
        return SyntheticSplitResult(
            train_indices=train_indices,
            test_indices=test_indices,
            cache_path=cache_path,
            from_cache=True,
        )

    if sample_labels is None:
        raise ValueError("sample_labels must be provided when synthetic split cache is missing")
    normalized_labels = _normalized_sample_labels(sample_labels)
    if len(normalized_labels) != sample_count:
        raise ValueError("sample_labels length must match sample_count")

    test_indices = _build_balanced_test_indices(
        sample_labels=normalized_labels,
        test_count=test_count,
        seed=seed,
    )
    test_set = set(test_indices)
    train_indices = [idx for idx in range(sample_count) if idx not in test_set]
    payload = {
        "dataset_name": dataset_name,
        "sample_count": sample_count,
        "test_size": float(test_size),
        "seed": int(seed),
        "dataset_fingerprint": dataset_fingerprint,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return SyntheticSplitResult(
        train_indices=train_indices,
        test_indices=test_indices,
        cache_path=cache_path,
        from_cache=False,
    )
