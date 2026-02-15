from __future__ import annotations

from dataclasses import replace

from app.models.entities import Detection

_TRIM_CHARS = " \t\r\n\"'`.,:;!?()[]{}<>"
_SENTENCE_STOP_CHARS = "\n\r.!?;"
_LOCATION_MARKERS = (
    "address",
    "street",
    "адрес",
    "улиц",
    "просп",
    "шоссе",
    "манзил",
    "ko'chasi",
    "mavze",
    "xonadon",
    "uy",
    "квартир",
    "дом",
    "apartment",
    "apt",
    "house",
)


def _trim_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max(0, int(start))
    right = min(len(text), int(end))
    while left < right and text[left] in _TRIM_CHARS:
        left += 1
    while right > left and text[right - 1] in _TRIM_CHARS:
        right -= 1
    return left, right


def _expand_until_sentence_boundary(
    *,
    text: str,
    start: int,
    end: int,
    max_expansion_chars: int,
) -> tuple[int, int]:
    left_limit = max(0, start - max_expansion_chars)
    right_limit = min(len(text), end + max_expansion_chars)

    left = start
    while left > left_limit and text[left - 1] not in _SENTENCE_STOP_CHARS:
        left -= 1

    right = end
    while right < right_limit and text[right] not in _SENTENCE_STOP_CHARS:
        right += 1

    return _trim_bounds(text, left, right)


def _looks_like_address(text_segment: str) -> bool:
    lowered = text_segment.lower()
    comma_count = text_segment.count(",")
    digit_count = sum(char.isdigit() for char in text_segment)
    marker_hits = sum(1 for marker in _LOCATION_MARKERS if marker in lowered)
    if marker_hits == 0:
        return False
    if marker_hits >= 1 and (comma_count >= 1 or digit_count >= 1):
        return True
    return marker_hits >= 2


def _normalize_location(
    *,
    text: str,
    detection: Detection,
    max_expansion_chars: int,
) -> Detection:
    start, end = _trim_bounds(text, detection.start, detection.end)
    if end <= start:
        return detection

    original_text = text[start:end]
    expanded_start, expanded_end = _expand_until_sentence_boundary(
        text=text,
        start=start,
        end=end,
        max_expansion_chars=max_expansion_chars,
    )
    if expanded_end <= expanded_start:
        return replace(detection, start=start, end=end, text=original_text)

    expanded_text = text[expanded_start:expanded_end]
    expanded_delta = (expanded_end - expanded_start) - (end - start)
    if expanded_delta <= 0:
        return replace(detection, start=start, end=end, text=original_text)
    if expanded_delta > max_expansion_chars:
        return replace(detection, start=start, end=end, text=original_text)
    if not _looks_like_address(expanded_text):
        return replace(detection, start=start, end=end, text=original_text)

    return replace(detection, start=expanded_start, end=expanded_end, text=expanded_text)


def _normalize_identifier(*, text: str, detection: Detection) -> Detection:
    start, end = _trim_bounds(text, detection.start, detection.end)
    if end <= start:
        return detection
    normalized_text = text[start:end]
    return replace(detection, start=start, end=end, text=normalized_text)


def _normalize_generic(*, text: str, detection: Detection) -> Detection:
    start, end = _trim_bounds(text, detection.start, detection.end)
    if end <= start:
        return detection
    return replace(detection, start=start, end=end, text=text[start:end])


def _resolve_overlaps(detections: list[Detection]) -> list[Detection]:
    if not detections:
        return []
    sorted_by_priority = sorted(
        detections,
        key=lambda item: (-float(item.score), -(item.end - item.start), item.start),
    )
    selected: list[Detection] = []
    for candidate in sorted_by_priority:
        if candidate.end <= candidate.start:
            continue
        if any(not (candidate.end <= item.start or candidate.start >= item.end) for item in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item.start)


def normalize_detections(
    *,
    text: str,
    detections: list[Detection],
    postprocess_config: dict | None,
) -> list[Detection]:
    if not detections:
        return detections
    cfg = postprocess_config or {}
    boundary_cfg = cfg.get("boundary", {}) if isinstance(cfg, dict) else {}
    if not isinstance(boundary_cfg, dict):
        return detections
    if not bool(boundary_cfg.get("enabled", False)):
        return detections

    max_expansion_chars = int(boundary_cfg.get("max_expansion_chars", 180))
    normalize_location_enabled = bool(boundary_cfg.get("location_enabled", True))
    normalize_identifier_enabled = bool(boundary_cfg.get("identifier_enabled", True))

    normalized: list[Detection] = []
    for item in detections:
        canonical = str(item.metadata.get("canonical_label") or "").lower()
        if canonical == "location" and normalize_location_enabled:
            normalized.append(
                _normalize_location(
                    text=text,
                    detection=item,
                    max_expansion_chars=max_expansion_chars,
                )
            )
            continue
        if canonical == "identifier" and normalize_identifier_enabled:
            normalized.append(_normalize_identifier(text=text, detection=item))
            continue
        normalized.append(_normalize_generic(text=text, detection=item))

    return _resolve_overlaps(normalized)
