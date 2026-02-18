from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import regex as re

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
    "tumani",
    "tuman",
    "shahri",
    "shahar",
    "xonadon",
    "uy",
    "kvartira",
    "dom",
    "квартир",
    "дом",
    "apartment",
    "apt",
    "house",
)
_LOCATION_CONTEXT_WORDS = (
    "address",
    "адрес",
    "location",
    "район",
    "область",
    "город",
    "улиц",
    "дом",
    "квартир",
    "корпус",
    "подъезд",
    "манзил",
    "tumani",
    "tuman",
    "shahri",
    "shahar",
    "шахр",
    "туман",
    "street",
    "district",
    "city",
    "house",
    "apartment",
)
_CANONICAL_OVERLAP_PRIORITY = {
    "ip": 140,
    "payment_card": 130,
    "phone": 125,
    "email": 120,
    "secret": 115,
    "identifier": 100,
    "date": 95,
    "person": 90,
    "organization": 85,
    "location": 80,
}

_GAP_BRIDGE_CANONICALS = {"person", "organization", "location"}
_MAX_GAP_BRIDGE_CHARS = 4
_GAP_BRIDGE_CHARS = set(" \t\r\n-.,'\"`/:;()[]{}<>")


@dataclass(slots=True)
class NormalizationStats:
    trimmed_spans: int = 0
    expanded_spans: int = 0
    invalid_spans_dropped: int = 0
    overlaps_dropped: int = 0


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


def _is_address_segment(segment_text: str) -> bool:
    lowered = segment_text.lower().strip()
    if not lowered:
        return False
    if any(marker in lowered for marker in _LOCATION_MARKERS):
        return True
    if any(char.isdigit() for char in lowered):
        return True
    # Covers short administrative fragments like "г. Москва", "р-н Арбат".
    if re.search(r"(?i)\b(?:г\.?|город|р-н|район|ул\.?|улица|пр-кт|д\.?|кв\.?|city|st\.?|district)\b", lowered):
        return True
    return False


def _location_clause_bounds(*, text: str, start: int, end: int, max_expansion_chars: int) -> tuple[int, int]:
    left_limit = max(0, start - max_expansion_chars)
    right_limit = min(len(text), end + max_expansion_chars)

    left = start
    while left > left_limit and text[left - 1] not in _SENTENCE_STOP_CHARS:
        left -= 1

    right = end
    while right < right_limit and text[right] not in _SENTENCE_STOP_CHARS:
        right += 1
    return left, right


def _split_comma_segments(text: str, start: int, end: int) -> list[tuple[int, int]]:
    if end <= start:
        return []
    segments: list[tuple[int, int]] = []
    seg_start = start
    idx = start
    while idx < end:
        if text[idx] == ",":
            segments.append((seg_start, idx))
            seg_start = idx + 1
        idx += 1
    segments.append((seg_start, end))
    return segments


def _expand_location_comma_chain(
    *,
    text: str,
    start: int,
    end: int,
    max_expansion_chars: int,
) -> tuple[int, int]:
    clause_start, clause_end = _location_clause_bounds(
        text=text,
        start=start,
        end=end,
        max_expansion_chars=max_expansion_chars,
    )
    segments = _split_comma_segments(text, clause_start, clause_end)
    if not segments:
        return start, end

    first_idx = None
    last_idx = None
    for idx, (seg_start, seg_end) in enumerate(segments):
        if seg_end <= start or seg_start >= end:
            continue
        if first_idx is None:
            first_idx = idx
        last_idx = idx
    if first_idx is None or last_idx is None:
        return start, end

    selected_start = start
    selected_end = end
    remaining_chars = max_expansion_chars

    current = last_idx + 1
    while current < len(segments) and remaining_chars > 0:
        seg_start, seg_end = segments[current]
        segment_text = text[seg_start:seg_end]
        if not _is_address_segment(segment_text):
            break
        new_start = selected_start
        new_end = seg_end
        growth = (new_end - new_start) - (selected_end - selected_start)
        if growth > remaining_chars:
            break
        selected_end = new_end
        remaining_chars -= max(0, growth)
        current += 1

    current = first_idx - 1
    while current >= 0 and remaining_chars > 0:
        seg_start, seg_end = segments[current]
        segment_text = text[seg_start:seg_end]
        if not _is_address_segment(segment_text):
            break
        new_start = seg_start
        new_end = selected_end
        growth = (new_end - new_start) - (selected_end - selected_start)
        if growth > remaining_chars:
            break
        selected_start = new_start
        remaining_chars -= max(0, growth)
        current -= 1

    return _trim_bounds(text, selected_start, selected_end)


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
    around_start = max(0, start - 24)
    around_end = min(len(text), end + 24)
    context_window = text[around_start:around_end].lower()
    has_context_signal = any(marker in context_window for marker in _LOCATION_CONTEXT_WORDS)
    if not has_context_signal and not _looks_like_address(original_text):
        return replace(detection, start=start, end=end, text=original_text)

    expanded_start, expanded_end = _expand_location_comma_chain(
        text=text,
        start=start,
        end=end,
        max_expansion_chars=max_expansion_chars,
    )
    expanded_text = text[expanded_start:expanded_end]
    if expanded_end <= expanded_start:
        return replace(detection, start=start, end=end, text=original_text)
    if (expanded_end - expanded_start) <= (end - start):
        return replace(detection, start=start, end=end, text=original_text)
    if not _is_address_segment(expanded_text):
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


def _resolve_overlaps(detections: list[Detection]) -> tuple[list[Detection], int]:
    # Deprecated: overlap-dropping is unsafe for leak prevention.
    # Kept for API/backwards-compat imports; callers should use union merge.
    return list(detections), 0


def _union_merge_spans(*, text: str, detections: list[Detection]) -> tuple[list[Detection], int]:
    """Merge overlapping/adjacent detections into non-overlapping mask spans.

    This is deliberately conservative: we never drop coverage, we only expand by union.
    """

    if not detections:
        return [], 0

    def rank(item: Detection) -> tuple[int, float, int, int]:
        metadata = item.metadata or {}
        canonical = str(metadata.get("canonical_label") or "").lower()
        base = int(_CANONICAL_OVERLAP_PRIORITY.get(canonical, 70))
        return (base, float(item.score), int(item.end - item.start), -int(item.start))

    items = [d for d in detections if int(d.end) > int(d.start)]
    items.sort(key=lambda d: (int(d.start), int(d.end)))

    merged: list[Detection] = []
    merges_performed = 0

    group: list[Detection] = []
    group_start = -1
    group_end = -1
    group_canonicals: set[str] = set()

    def canonical(item: Detection) -> str:
        meta = item.metadata or {}
        return str(meta.get("canonical_label") or "").lower().strip()

    def gap_is_bridgeable(*, gap_text: str) -> bool:
        if not gap_text:
            return False
        if len(gap_text) > _MAX_GAP_BRIDGE_CHARS:
            return False
        return all(ch in _GAP_BRIDGE_CHARS for ch in gap_text)

    def flush() -> None:
        nonlocal merges_performed, group, group_start, group_end, group_canonicals
        if not group:
            return
        best = max(group, key=rank)
        max_score = max(float(item.score) for item in group)
        detectors = sorted({str(item.detector or "") for item in group if str(item.detector or "").strip()})
        canonicals = sorted(
            {str((item.metadata or {}).get("canonical_label") or "").lower() for item in group if (item.metadata or {}).get("canonical_label")}
        )
        meta = dict(best.metadata or {})
        meta["merged_from_detectors"] = detectors
        meta["merged_from_canonical_labels"] = canonicals
        merged.append(
            Detection(
                start=int(group_start),
                end=int(group_end),
                text=text[int(group_start) : int(group_end)],
                label=str(best.label),
                score=float(max_score),
                detector=str(best.detector),
                metadata=meta,
            )
        )
        merges_performed += max(0, len(group) - 1)
        group = []
        group_start = -1
        group_end = -1
        group_canonicals = set()

    for item in items:
        start = int(item.start)
        end = int(item.end)
        item_canonical = canonical(item)
        if not group:
            group = [item]
            group_start = start
            group_end = end
            if item_canonical:
                group_canonicals = {item_canonical}
            continue
        if start <= group_end:  # overlap or adjacency
            group.append(item)
            group_end = max(group_end, end)
            group_start = min(group_start, start)
            if item_canonical:
                group_canonicals.add(item_canonical)
            continue

        gap = text[int(group_end) : int(start)]
        if (
            item_canonical
            and item_canonical in _GAP_BRIDGE_CANONICALS
            and item_canonical in group_canonicals
            and gap_is_bridgeable(gap_text=gap)
        ):
            # Bridge small delimiter gaps (spaces/punct) for the same canonical class to avoid
            # fragmented names/orgs/locations hurting char-level recall.
            group.append(item)
            group_end = max(group_end, end)
            group_start = min(group_start, start)
            group_canonicals.add(item_canonical)
            continue

        flush()
        group = [item]
        group_start = start
        group_end = end
        group_canonicals = {item_canonical} if item_canonical else set()

    flush()
    return merged, merges_performed


def normalize_detections(
    *,
    text: str,
    detections: list[Detection],
    return_stats: bool = False,
) -> list[Detection] | tuple[list[Detection], dict[str, int]]:
    if not detections:
        empty_stats = {
            "trimmed_spans": 0,
            "expanded_spans": 0,
            "invalid_spans_dropped": 0,
            "overlaps_dropped": 0,
        }
        return (detections, empty_stats) if return_stats else detections
    max_expansion_chars = 180
    normalize_location_enabled = True
    normalize_identifier_enabled = True

    normalized: list[Detection] = []
    stats = NormalizationStats()
    for item in detections:
        canonical = str(item.metadata.get("canonical_label") or "").lower()
        updated = item
        if canonical == "location" and normalize_location_enabled:
            updated = _normalize_location(
                text=text,
                detection=item,
                max_expansion_chars=max_expansion_chars,
            )
        elif canonical == "identifier" and normalize_identifier_enabled:
            updated = _normalize_identifier(text=text, detection=item)
        else:
            updated = _normalize_generic(text=text, detection=item)

        if updated.end <= updated.start:
            stats.invalid_spans_dropped += 1
            continue

        old_len = item.end - item.start
        new_len = updated.end - updated.start
        if old_len > 0 and new_len > old_len:
            stats.expanded_spans += 1
        elif old_len > 0 and new_len < old_len:
            stats.trimmed_spans += 1

        normalized.append(updated)

    resolved, merges_performed = _union_merge_spans(text=text, detections=normalized)
    stats.overlaps_dropped = merges_performed
    payload = {
        "trimmed_spans": stats.trimmed_spans,
        "expanded_spans": stats.expanded_spans,
        "invalid_spans_dropped": stats.invalid_spans_dropped,
        "overlaps_dropped": stats.overlaps_dropped,
    }

    if return_stats:
        return resolved, payload
    return resolved
