from __future__ import annotations

from dataclasses import dataclass
from typing import DefaultDict

from app.eval.types import EvalSample, EvalSpan


@dataclass(frozen=True, slots=True)
class ScoredSpansView:
    samples: list[EvalSample]
    predictions_by_id: dict[str, list[EvalSpan]]
    unscored_predictions_by_label: dict[str, int]


def filter_scored_spans(
    *,
    samples: list[EvalSample],
    predictions_by_id: dict[str, list[EvalSpan]],
    scored_labels: set[str] | frozenset[str],
) -> ScoredSpansView:
    scored_set = {str(item) for item in scored_labels}

    # Gold: only spans with canonical labels in scope.
    filtered_samples: list[EvalSample] = []
    for sample in samples:
        gold_scored = [
            span
            for span in sample.gold_spans
            if span.canonical_label is not None and span.canonical_label in scored_set
        ]
        filtered_samples.append(
            EvalSample(
                sample_id=sample.sample_id,
                text=sample.text,
                gold_spans=gold_scored,
                metadata=dict(sample.metadata),
            )
        )

    # Predictions: include only scored labels for metrics; count other canonical labels as unscored.
    unscored: DefaultDict[str, int] = DefaultDict(int)
    filtered_predictions: dict[str, list[EvalSpan]] = {}
    for sample_id, spans in predictions_by_id.items():
        scored: list[EvalSpan] = []
        for span in spans:
            canonical = span.canonical_label
            if canonical is None:
                continue
            if canonical in scored_set:
                scored.append(span)
            else:
                unscored[str(canonical)] += 1
        filtered_predictions[sample_id] = scored

    return ScoredSpansView(
        samples=filtered_samples,
        predictions_by_id=filtered_predictions,
        unscored_predictions_by_label=dict(unscored),
    )

