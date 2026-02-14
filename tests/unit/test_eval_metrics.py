from __future__ import annotations

from app.eval.metrics import evaluate_samples, match_counts
from app.eval.types import EvalSample, EvalSpan


def test_match_counts_exact_vs_overlap() -> None:
    gold = [EvalSpan(start=10, end=20, label="A")]
    pred_overlap = [EvalSpan(start=12, end=18, label="B")]

    exact = match_counts(gold, pred_overlap, require_label=False, allow_overlap=False)
    overlap = match_counts(gold, pred_overlap, require_label=False, allow_overlap=True)

    assert exact.true_positives == 0
    assert exact.false_positives == 1
    assert exact.false_negatives == 1

    assert overlap.true_positives == 1
    assert overlap.false_positives == 0
    assert overlap.false_negatives == 0


def test_evaluate_samples_label_aware() -> None:
    sample = EvalSample(
        sample_id="s1",
        text="text",
        gold_spans=[EvalSpan(start=0, end=4, label="gold_person", canonical_label="person")],
    )
    predictions = {
        "s1": [EvalSpan(start=0, end=4, label="pred_person", canonical_label="person")],
    }

    result = evaluate_samples([sample], predictions)
    assert result.exact_agnostic.true_positives == 1
    assert result.exact_canonical.true_positives == 1
    assert result.per_label_exact["person"].true_positives == 1
