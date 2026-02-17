from __future__ import annotations

from app.eval.run import _load_and_select_datasets
from app.eval.suite_loader import DatasetSpec, SuiteSpec


def test_tag_selection_is_and_semantics() -> None:
    suite = SuiteSpec(
        suite_id="s",
        default_collection="",
        default_split="fast",
        scored_labels=("person",),
        datasets=(
            DatasetSpec(
                dataset_id="a",
                format="privacy_mask_parquet_v1",
                text_field="source_text",
                mask_field="privacy_mask",
                annotated_labels=("person",),
                gold_label_mapping={"person": "person"},
                slice_fields=tuple(),
                tags=("en", "stress"),
                notes="",
            ),
            DatasetSpec(
                dataset_id="b",
                format="privacy_mask_parquet_v1",
                text_field="source_text",
                mask_field="privacy_mask",
                annotated_labels=("person",),
                gold_label_mapping={"person": "person"},
                slice_fields=tuple(),
                tags=("en",),
                notes="",
            ),
        ),
    )

    selected = _load_and_select_datasets(
        suite=suite,
        collection_ids=None,
        requested_datasets=None,
        requested_tags=["en", "stress"],
    )
    assert [d.dataset_id for d in selected] == ["a"]

