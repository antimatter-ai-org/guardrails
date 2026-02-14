from __future__ import annotations

import math
from collections import Counter

import regex as re

from app.detectors.base import Detector
from app.models.entities import Detection


class EntropyDetector(Detector):
    def __init__(
        self,
        name: str,
        min_length: int = 20,
        entropy_threshold: float = 3.6,
        pattern: str = r"\b[A-Za-z0-9_\-/+=]{20,}\b",
    ) -> None:
        super().__init__(name)
        self._min_length = min_length
        self._entropy_threshold = entropy_threshold
        self._regex = re.compile(pattern)

    @staticmethod
    def _shannon_entropy(value: str) -> float:
        if not value:
            return 0.0
        counts = Counter(value)
        length = len(value)
        entropy = 0.0
        for count in counts.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        return entropy

    def detect(self, text: str) -> list[Detection]:
        findings: list[Detection] = []
        for match in self._regex.finditer(text):
            token = match.group(0)
            if len(token) < self._min_length:
                continue
            entropy = self._shannon_entropy(token)
            if entropy < self._entropy_threshold:
                continue
            findings.append(
                Detection(
                    start=match.start(),
                    end=match.end(),
                    text=token,
                    label="SECRET_HIGH_ENTROPY",
                    score=min(0.99, 0.7 + (entropy - self._entropy_threshold) / 4),
                    detector=self.name,
                    metadata={"entropy": round(entropy, 4)},
                )
            )
        return findings
