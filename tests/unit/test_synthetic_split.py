from __future__ import annotations

from pathlib import Path

import pytest

from app.eval.datasets.synthetic_split import load_or_create_synthetic_split


def _label_prevalence(sample_labels: list[set[str]], indices: list[int], label: str) -> float:
    if not indices:
        return 0.0
    hits = 0
    for idx in indices:
        if label in sample_labels[idx]:
            hits += 1
    return hits / len(indices)


def test_synthetic_split_is_cached_and_deterministic(tmp_path: Path) -> None:
    sample_labels: list[set[str]] = []
    sample_labels.extend([{"person"} for _ in range(60)])
    sample_labels.extend([{"email"} for _ in range(30)])
    sample_labels.extend([{"phone"} for _ in range(10)])

    first = load_or_create_synthetic_split(
        dataset_name="demo/synthetic",
        cache_dir=str(tmp_path),
        sample_count=len(sample_labels),
        sample_labels=sample_labels,
        test_size=0.2,
        seed=42,
        dataset_fingerprint="fp1",
    )
    second = load_or_create_synthetic_split(
        dataset_name="demo/synthetic",
        cache_dir=str(tmp_path),
        sample_count=len(sample_labels),
        test_size=0.2,
        seed=42,
        dataset_fingerprint="fp1",
    )

    assert first.from_cache is False
    assert second.from_cache is True
    assert first.cache_path.exists()
    assert first.test_indices == second.test_indices
    assert first.train_indices == second.train_indices
    assert len(first.test_indices) == 20
    assert len(first.train_indices) == 80


def test_synthetic_split_preserves_label_distribution_reasonably(tmp_path: Path) -> None:
    sample_labels: list[set[str]] = []
    sample_labels.extend([{"person", "identifier"} for _ in range(40)])
    sample_labels.extend([{"email"} for _ in range(30)])
    sample_labels.extend([{"phone"} for _ in range(20)])
    sample_labels.extend([set() for _ in range(10)])

    split = load_or_create_synthetic_split(
        dataset_name="demo/balanced",
        cache_dir=str(tmp_path),
        sample_count=len(sample_labels),
        sample_labels=sample_labels,
        test_size=0.25,
        seed=7,
        dataset_fingerprint="fp2",
    )

    for label in ("person", "identifier", "email", "phone"):
        full = _label_prevalence(sample_labels, list(range(len(sample_labels))), label)
        test = _label_prevalence(sample_labels, split.test_indices, label)
        assert abs(full - test) <= 0.12


def test_synthetic_split_requires_labels_when_cache_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_or_create_synthetic_split(
            dataset_name="demo/missing",
            cache_dir=str(tmp_path),
            sample_count=10,
            test_size=0.2,
            seed=11,
            dataset_fingerprint="fp3",
        )
