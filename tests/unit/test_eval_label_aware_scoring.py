from __future__ import annotations

from app.eval.metrics import evaluate_samples
from app.eval.types import EvalSample, EvalSpan, MetricCounts
from app.eval.metrics.aggregation import macro_over_labels
from app.eval.metrics.risk_weighting import risk_weighted_char_recall
from app.eval.metrics.spans import filter_scored_spans


def test_filter_scored_spans_ignores_unlabeled_predictions_but_counts_them() -> None:
    sample = EvalSample(
        sample_id="s1",
        text="a b c d",
        gold_spans=[EvalSpan(start=0, end=1, label="PERSON", canonical_label="person")],
        metadata={},
    )
    predictions = {
        "s1": [
            EvalSpan(start=0, end=1, label="PERSON", canonical_label="person"),
            EvalSpan(start=2, end=3, label="IP_ADDRESS", canonical_label="ip"),
        ]
    }

    view = filter_scored_spans(samples=[sample], predictions_by_id=predictions, scored_labels={"person"})
    assert view.unscored_predictions_by_label == {"ip": 1}

    agg = evaluate_samples(view.samples, view.predictions_by_id)
    assert agg.exact_canonical.false_positives == 0
    assert "person" in agg.per_label_exact
    assert "ip" not in agg.per_label_exact


def test_macro_over_labels_excludes_labels_without_gold_support() -> None:
    per_label = {
        "person": MetricCounts(true_positives=1, false_positives=0, false_negatives=1),
        "ip": MetricCounts(true_positives=0, false_positives=5, false_negatives=0),  # no gold support
    }
    macro = macro_over_labels(per_label)
    assert macro.labels_included == 1


def test_risk_weighted_char_recall_uses_weights_and_gold_support() -> None:
    per_label_char = {
        "secret": MetricCounts(true_positives=5, false_positives=0, false_negatives=5),  # recall 0.5, weight 5
        "date": MetricCounts(true_positives=9, false_positives=0, false_negatives=1),  # recall 0.9, weight 1
        "ip": MetricCounts(true_positives=0, false_positives=0, false_negatives=0),  # excluded
    }
    out = risk_weighted_char_recall(per_label_char=per_label_char)
    # Weighted average = (0.5*5 + 0.9*1) / 6 = 0.566666...
    assert abs(out.value - (0.5 * 5 + 0.9 * 1) / 6) < 1e-9
    assert out.labels_included == 2

