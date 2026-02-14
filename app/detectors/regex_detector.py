from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import regex as re

from app.detectors.base import Detector
from app.models.entities import Detection

_FLAG_MAP: dict[str, int] = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "UNICODE": re.UNICODE,
}


@dataclass(slots=True)
class CompiledPattern:
    name: str
    label: str
    regex: re.Pattern
    score: float


class RegexDetector(Detector):
    def __init__(self, name: str, patterns: list[dict[str, Any]]) -> None:
        super().__init__(name)
        compiled: list[CompiledPattern] = []
        for pattern in patterns:
            flags_value = 0
            for flag in pattern.get("flags", []):
                flags_value |= _FLAG_MAP.get(str(flag).upper(), 0)
            compiled.append(
                CompiledPattern(
                    name=pattern["name"],
                    label=pattern["label"],
                    regex=re.compile(pattern["pattern"], flags_value),
                    score=float(pattern.get("score", 0.8)),
                )
            )
        self._patterns = compiled

    def detect(self, text: str) -> list[Detection]:
        findings: list[Detection] = []
        for pattern in self._patterns:
            for match in pattern.regex.finditer(text):
                findings.append(
                    Detection(
                        start=match.start(),
                        end=match.end(),
                        text=match.group(0),
                        label=pattern.label,
                        score=pattern.score,
                        detector=self.name,
                        metadata={"pattern": pattern.name},
                    )
                )
        return findings
