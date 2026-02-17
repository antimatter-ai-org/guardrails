from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.eval.types import MetricCounts


def metric_payload(counts: MetricCounts) -> dict[str, Any]:
    return {
        "true_positives": counts.true_positives,
        "false_positives": counts.false_positives,
        "false_negatives": counts.false_negatives,
        "precision": round(counts.precision, 6),
        "recall": round(counts.recall, 6),
        "f1": round(counts.f1, 6),
        "residual_miss_ratio": round(1.0 - counts.recall, 6),
        "support_pred": counts.true_positives + counts.false_positives,
        "support_gold": counts.true_positives + counts.false_negatives,
    }


@dataclass(frozen=True, slots=True)
class WeightedMacroResult:
    value: float
    covered_labels: tuple[str, ...]
    skipped_labels: tuple[str, ...]
    total_weight: float


def weighted_macro_recall(
    *,
    per_label_counts: dict[str, MetricCounts],
    weights: dict[str, float],
) -> WeightedMacroResult:
    numer = 0.0
    denom = 0.0
    covered: list[str] = []
    skipped: list[str] = []

    for label, counts in sorted(per_label_counts.items()):
        gold_support = counts.true_positives + counts.false_negatives
        if gold_support <= 0:
            skipped.append(label)
            continue
        w = float(weights.get(label, 1.0))
        denom += w
        numer += w * float(counts.recall)
        covered.append(label)

    value = numer / denom if denom > 0 else 0.0
    return WeightedMacroResult(
        value=round(value, 6),
        covered_labels=tuple(covered),
        skipped_labels=tuple(skipped),
        total_weight=round(denom, 6),
    )

