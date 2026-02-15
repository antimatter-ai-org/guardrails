from __future__ import annotations

from app.eval.datasets.registry import get_dataset_adapter, list_supported_datasets


def test_supported_datasets_include_scanpatch_and_rubai() -> None:
    names = list_supported_datasets()
    assert "scanpatch/pii-ner-corpus-synthetic-controlled" in names
    assert "BoburAmirov/rubai-NER-150K-Personal" in names


def test_get_dataset_adapter_returns_matching_dataset_name() -> None:
    for name in list_supported_datasets():
        adapter = get_dataset_adapter(name)
        assert adapter.dataset_name == name
