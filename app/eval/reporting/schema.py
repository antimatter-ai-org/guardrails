from __future__ import annotations

from typing import Any

from app.eval.types import MetricCounts
from app.eval.metrics import EvaluationAggregate
from app.eval.report import metrics_payload
from app.eval.metrics.classification import BinaryCounts


REPORT_VERSION = "3.0"


def metric_counts_payload(metric: MetricCounts) -> dict[str, Any]:
    return {
        "true_positives": metric.true_positives,
        "false_positives": metric.false_positives,
        "false_negatives": metric.false_negatives,
        "precision": round(metric.precision, 6),
        "recall": round(metric.recall, 6),
        "f1": round(metric.f1, 6),
        "residual_miss_ratio": round(1.0 - metric.recall, 6),
    }


def aggregate_payload(aggregate: EvaluationAggregate) -> dict[str, Any]:
    # Reuse the v2 payload shape for span metrics for compatibility with existing tooling.
    return metrics_payload(aggregate)


def binary_counts_payload(counts: BinaryCounts) -> dict[str, Any]:
    return {
        "tp": counts.tp,
        "fp": counts.fp,
        "tn": counts.tn,
        "fn": counts.fn,
        "precision": round(counts.precision, 6),
        "recall": round(counts.recall, 6),
        "f1": round(counts.f1, 6),
        "false_positive_rate": round(counts.false_positive_rate, 6),
        "false_negative_rate": round(counts.false_negative_rate, 6),
    }

