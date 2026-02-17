from __future__ import annotations

from app.eval.cache_paths import EvalCachePaths
from app.eval.suite_loader import DatasetSpec
from app.eval.views import ViewSpec, parse_where_clauses, resolve_view_indices


def test_view_cache_is_deterministic(tmp_path) -> None:
    cache = EvalCachePaths.from_cache_dir_arg(str(tmp_path / ".eval_cache"))
    dataset_rows = [
        {"language": "en", "privacy_mask": []},
        {"language": "ru", "privacy_mask": [{"label": "person"}]},
        {"language": "en", "privacy_mask": [{"label": "person"}]},
    ]
    spec = DatasetSpec(
        dataset_id="org/ds",
        format="privacy_mask_parquet_v1",
        text_field="source_text",
        mask_field="privacy_mask",
        annotated_labels=("person",),
        gold_label_mapping={"person": "person"},
        slice_fields=tuple(),
        tags=tuple(),
        notes="",
    )

    def scored_count(row):
        return 1 if row.get("privacy_mask") else 0

    view = ViewSpec(
        base_split="fast",
        where=parse_where_clauses(["language=en"]),
        max_samples=2,
        seed=123,
        stratify_by=tuple(),
        view_name=None,
    )

    idx1, meta1 = resolve_view_indices(
        dataset_rows=dataset_rows,
        dataset_id="org/ds",
        dataset_fingerprint="fp1",
        spec=spec,
        cache_paths=cache,
        view=view,
        scored_entity_count_fn=scored_count,
        label_presence_fn=None,
    )
    idx2, meta2 = resolve_view_indices(
        dataset_rows=dataset_rows,
        dataset_id="org/ds",
        dataset_fingerprint="fp1",
        spec=spec,
        cache_paths=cache,
        view=view,
        scored_entity_count_fn=scored_count,
        label_presence_fn=None,
    )

    assert idx1 == idx2
    assert meta2["from_cache"] is True

