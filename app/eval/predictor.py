from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from app.config import PolicyDefinition
from app.detectors.base import Detector
from app.eval.labels import canonicalize_prediction_label
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


def _merge_priority(item: Detection) -> float:
    priority = item.score
    detector_name = item.detector.lower()
    label = canonicalize_prediction_label(item.label)

    if detector_name.endswith("_regex") or "regex" in detector_name:
        priority += 0.15

    # Keep deterministic temporal/network/id detections when they overlap with generic NER.
    if label in {"ip", "date", "identifier", "email", "payment_card"}:
        priority += 0.1
    return priority


def _resolve_overlaps_weighted(detections: list[Detection]) -> list[Detection]:
    if not detections:
        return []

    sorted_by_priority = sorted(
        detections,
        key=lambda item: (-_merge_priority(item), -(item.end - item.start), item.start),
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
    return _resolve_overlaps_weighted(filtered)


def detect_only(
    text: str,
    policy: PolicyDefinition,
    detectors: Iterable[Detector],
) -> list[Detection]:
    detections: list[Detection] = []
    for detector in detectors:
        detections.extend(detector.detect(text))
    filtered = [item for item in detections if item.end > item.start and item.score >= policy.min_score]
    return _resolve_overlaps_weighted(filtered)


def sample_uncertainty_score(text: str, stage_a_detections: list[Detection], min_score: float) -> float:
    if not stage_a_detections:
        return 1.0

    score = 0.0
    canonical_counts = Counter(canonicalize_prediction_label(item.label) for item in stage_a_detections)
    low_conf = sum(1 for item in stage_a_detections if item.score <= min_score + 0.1)
    if low_conf:
        score += min(0.35, 0.08 * low_conf)

    phone_hits = canonical_counts.get("phone", 0)
    if phone_hits >= 2:
        score += min(0.4, 0.1 * phone_hits)

    text_lower = text.lower()
    address_hint = any(
        marker in text_lower
        for marker in (" улиц", "street", "avenue", "адрес", "квартира", "дом ", "район", "просп")
    )
    if address_hint and canonical_counts.get("location", 0) == 0:
        score += 0.35

    digit_count = sum(ch.isdigit() for ch in text)
    if digit_count >= 10 and (
        canonical_counts.get("identifier", 0) == 0
        and canonical_counts.get("ip", 0) == 0
        and canonical_counts.get("phone", 0) == 0
    ):
        score += 0.35

    if len(stage_a_detections) >= 5 and canonical_counts.get("person", 0) >= 4:
        score += 0.2

    return min(1.0, score)


def merge_cascade_detections(stage_a: list[Detection], stage_b: list[Detection]) -> list[Detection]:
    return _resolve_overlaps_weighted([*stage_a, *stage_b])
