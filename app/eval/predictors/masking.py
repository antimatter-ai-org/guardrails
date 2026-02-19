from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.masking.reversible import ReversibleMaskingEngine


@dataclass(frozen=True, slots=True)
class MaskResult:
    masked_text: str
    findings_count: int


def mask_text_with_detections(*, text: str, detections: list[Any], placeholder_prefix: str = "GR00") -> MaskResult:
    # For eval-only leakage diagnostics we don't need session storage; just deterministic masking.
    masker = ReversibleMaskingEngine(str(placeholder_prefix))
    masked = masker.mask(text, detections)
    return MaskResult(masked_text=masked.text, findings_count=len(masked.detections))

