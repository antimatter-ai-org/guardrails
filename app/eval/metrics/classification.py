from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BinaryCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return 0.0 if denom == 0 else self.tp / denom

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return 0.0 if denom == 0 else self.tp / denom

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return 0.0 if denom == 0 else self.fp / denom

    @property
    def false_negative_rate(self) -> float:
        denom = self.tp + self.fn
        return 0.0 if denom == 0 else self.fn / denom

