from __future__ import annotations

from app.eval.aggregate import weighted_macro_recall
from app.eval.types import MetricCounts


def test_weighted_macro_skips_labels_with_no_gold_support() -> None:
    per_label = {
        "person": MetricCounts(true_positives=8, false_positives=2, false_negatives=2),  # recall=0.8
        "secret": MetricCounts(true_positives=0, false_positives=0, false_negatives=0),  # no gold
    }
    weights = {"person": 2.0, "secret": 10.0}
    res = weighted_macro_recall(per_label_counts=per_label, weights=weights)
    assert res.covered_labels == ("person",)
    assert "secret" in res.skipped_labels
    assert res.value == 0.8

