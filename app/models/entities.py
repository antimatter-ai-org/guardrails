from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Detection:
    start: int
    end: int
    text: str
    label: str
    score: float
    detector: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MaskingResult:
    text: str
    placeholders: dict[str, str]
    detections: list[Detection]


@dataclass(slots=True)
class UnmaskingResult:
    text: str
    replaced: int
