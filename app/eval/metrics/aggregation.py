from __future__ import annotations

from dataclasses import dataclass

from app.eval.types import MetricCounts


@dataclass(frozen=True, slots=True)
class MacroMetric:
    precision: float
    recall: float
    f1: float
    labels_included: int


def _has_gold_support(counts: MetricCounts) -> bool:
    # gold support exists if there are any gold units: TP + FN
    return (counts.true_positives + counts.false_negatives) > 0


def macro_over_labels(per_label: dict[str, MetricCounts]) -> MacroMetric:
    items: list[MetricCounts] = [m for m in per_label.values() if _has_gold_support(m)]
    if not items:
        return MacroMetric(precision=0.0, recall=0.0, f1=0.0, labels_included=0)
    return MacroMetric(
        precision=sum(m.precision for m in items) / len(items),
        recall=sum(m.recall for m in items) / len(items),
        f1=sum(m.f1 for m in items) / len(items),
        labels_included=len(items),
    )

