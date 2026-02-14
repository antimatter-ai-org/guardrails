from __future__ import annotations

from app.eval.types import EvalSample, EvalSpan
from app.finetune.gliner_pipeline import (
    _char_to_word_span,
    build_training_bundle_from_eval_samples,
)


def test_char_to_word_span_exact_and_overlap() -> None:
    starts = [0, 6, 12]
    ends = [5, 11, 18]
    start_to_idx = {value: idx for idx, value in enumerate(starts)}
    end_to_idx = {value: idx for idx, value in enumerate(ends)}

    assert _char_to_word_span(0, 5, starts, ends, start_to_idx, end_to_idx) == (0, 0)
    assert _char_to_word_span(6, 11, starts, ends, start_to_idx, end_to_idx) == (1, 1)

    # Non-aligned span still resolves via overlap fallback.
    assert _char_to_word_span(7, 17, starts, ends, start_to_idx, end_to_idx) == (1, 2)


def test_build_training_bundle_maps_canonical_labels() -> None:
    text = "Alice email a@example.com"
    sample = EvalSample(
        sample_id="s1",
        text=text,
        gold_spans=[
            EvalSpan(start=0, end=5, label="name", canonical_label="person"),
            EvalSpan(start=12, end=25, label="email", canonical_label="email"),
            EvalSpan(start=6, end=11, label="drop", canonical_label=None),
        ],
    )

    bundle = build_training_bundle_from_eval_samples(
        dataset_name="dummy",
        splits=["train"],
        samples=[sample],
        splitter_type="whitespace",
    )

    assert bundle.stats["total_samples"] == 1
    assert bundle.stats["gold_spans_total"] == 3
    assert bundle.stats["gold_spans_mapped"] == 2
    assert bundle.stats["gold_spans_dropped_unmapped"] == 1
    assert bundle.labels == ["person", "email"]
    assert len(bundle.training_records) == 1
    assert bundle.training_records[0]["tokenized_text"] == ["Alice", "email", "a", "@", "example", ".", "com"]
    assert bundle.training_records[0]["ner"] == [(0, 0, "person"), (2, 6, "email")]
