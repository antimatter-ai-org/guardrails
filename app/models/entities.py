from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AnalysisDiagnostics:
    elapsed_ms: float
    detector_timing_ms: dict[str, float] = field(default_factory=dict)
    detector_span_counts: dict[str, int] = field(default_factory=dict)
    detector_errors: dict[str, str] = field(default_factory=dict)
    postprocess_mutations: dict[str, int] = field(default_factory=dict)
    limit_flags: dict[str, bool] = field(default_factory=dict)


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
