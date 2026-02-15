from __future__ import annotations

import sys
import types

from app.eval import run as eval_run


def test_resolve_dataset_split_strict_skips_when_missing(monkeypatch) -> None:
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.get_dataset_split_names = lambda dataset_name, token=None: ["train"]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    split, available = eval_run._resolve_dataset_split(
        dataset_name="BoburAmirov/rubai-NER-150K-Personal",
        requested_split="test",
        hf_token=None,
        strict_split=True,
    )

    assert split is None
    assert available == ["train"]


def test_resolve_dataset_split_non_strict_falls_back_to_train(monkeypatch) -> None:
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.get_dataset_split_names = lambda dataset_name, token=None: ["train"]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    split, available = eval_run._resolve_dataset_split(
        dataset_name="BoburAmirov/rubai-NER-150K-Personal",
        requested_split="test",
        hf_token=None,
        strict_split=False,
    )

    assert split == "train"
    assert available == ["train"]


def test_resolve_dataset_split_allows_synthetic_test_split(monkeypatch) -> None:
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.get_dataset_split_names = lambda dataset_name, token=None: ["train"]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    split, available = eval_run._resolve_dataset_split(
        dataset_name="BoburAmirov/rubai-NER-150K-Personal",
        requested_split="test",
        hf_token=None,
        strict_split=True,
        allow_synthetic_split=True,
    )

    assert split == "test"
    assert available == ["train"]
