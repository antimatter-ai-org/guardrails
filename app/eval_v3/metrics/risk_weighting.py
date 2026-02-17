from __future__ import annotations

from dataclasses import dataclass

from app.eval.types import MetricCounts
from app.eval_v3.taxonomy import DEFAULT_RISK_WEIGHTS, WeightedValue, weighted_average


@dataclass(frozen=True, slots=True)
class RiskWeightedRecall:
    value: float
    labels_included: int
    total_weight: float


def risk_weighted_char_recall(
    *,
    per_label_char: dict[str, MetricCounts],
    risk_weights: dict[str, float] | None = None,
) -> RiskWeightedRecall:
    weights = risk_weights or DEFAULT_RISK_WEIGHTS
    items: list[WeightedValue] = []
    for label, counts in per_label_char.items():
        gold = counts.true_positives + counts.false_negatives
        if gold <= 0:
            continue
        w = float(weights.get(label, 0.0))
        if w <= 0:
            continue
        items.append(WeightedValue(value=counts.recall, weight=w))
    total_w = sum(item.weight for item in items)
    return RiskWeightedRecall(
        value=weighted_average(items),
        labels_included=len(items),
        total_weight=total_w,
    )

