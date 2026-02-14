from __future__ import annotations

from app.config import PolicyDefinition
from app.detectors.base import Detector
from app.models.entities import Detection


def _resolve_overlaps(detections: list[Detection]) -> list[Detection]:
    if not detections:
        return []

    sorted_by_priority = sorted(
        detections,
        key=lambda item: (-item.score, -(item.end - item.start), item.start),
    )

    selected: list[Detection] = []
    for candidate in sorted_by_priority:
        if any(not (candidate.end <= existing.start or candidate.start >= existing.end) for existing in selected):
            continue
        selected.append(candidate)

    return sorted(selected, key=lambda item: item.start)


def detect_with_policy(
    text: str,
    policy: PolicyDefinition,
    detectors: list[Detector],
) -> list[Detection]:
    detections: list[Detection] = []
    for detector in detectors:
        detections.extend(detector.detect(text))

    filtered = [item for item in detections if item.end > item.start and item.score >= policy.min_score]
    return _resolve_overlaps(filtered)
