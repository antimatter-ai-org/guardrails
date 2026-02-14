from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass(slots=True)
class EvalSpan:
    start: int
    end: int
    label: str
    canonical_label: str | None = None
    score: float | None = None
    detector: str | None = None


@dataclass(slots=True)
class EvalSample:
    sample_id: str
    text: str
    gold_spans: list[EvalSpan]
    metadata: dict[str, str | bool | int | float | None] = field(default_factory=dict)


@dataclass(slots=True)
class MetricCounts:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
