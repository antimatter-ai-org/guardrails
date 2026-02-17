from __future__ import annotations

from app.eval.metrics import evaluate_samples
from app.eval.types import EvalSample, EvalSpan


def test_allowed_labels_filters_predictions_and_gold() -> None:
    sample = EvalSample(
        sample_id="s1",
        text="hello",
        gold_spans=[EvalSpan(start=0, end=5, label="PERSON", canonical_label="person")],
    )
    predictions = {
        "s1": [
            EvalSpan(start=0, end=5, label="PERSON", canonical_label="person"),
            # Dataset doesn't annotate email, so the evaluator should ignore it if allowed_labels excludes it.
            EvalSpan(start=0, end=5, label="EMAIL", canonical_label="email"),
        ]
    }

    agg = evaluate_samples([sample], predictions, allowed_labels={"person"})
    assert agg.exact_canonical.false_positives == 0
    assert set(agg.per_label_exact.keys()) == {"person"}

