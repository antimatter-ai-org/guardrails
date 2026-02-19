from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(slots=True, frozen=True)
class TextChunkWindow:
    text_start: int
    text_end: int


def deterministic_overlap_tokens(max_input_tokens: int) -> int:
    """Deterministic overlap derived from the model context length.

    This is intentionally not configurable at runtime to avoid hidden caps and
    variability in coverage.
    """

    value = max(16, int(max_input_tokens) // 8)
    return max(0, min(value, max(0, int(max_input_tokens) - 1)))


def _require_fast_tokenizer(tokenizer: Any) -> None:
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("fast tokenizer is required (need offsets_mapping support)")


def _coerce_offset_chunks(offset_mapping: Any) -> list[list[tuple[int, int]]]:
    # For a single input, tokenizers may return either:
    # - list[tuple[int,int]] (single window)
    # - list[list[tuple[int,int]]] (multiple windows)
    if not isinstance(offset_mapping, list) or not offset_mapping:
        return []
    first = offset_mapping[0]
    if isinstance(first, (tuple, list)) and len(first) == 2 and all(isinstance(x, int) for x in first):
        return [list(offset_mapping)]  # type: ignore[arg-type]
    if isinstance(first, list):
        out: list[list[tuple[int, int]]] = []
        for chunk in offset_mapping:
            if not isinstance(chunk, list):
                continue
            pairs: list[tuple[int, int]] = []
            for item in chunk:
                if (
                    isinstance(item, (tuple, list))
                    and len(item) == 2
                    and isinstance(item[0], int)
                    and isinstance(item[1], int)
                ):
                    pairs.append((int(item[0]), int(item[1])))
            out.append(pairs)
        return out
    return []


def chunk_text(
    *,
    text: str,
    tokenizer: Any,
    max_input_tokens: int,
    overlap_tokens: int | None = None,
) -> list[TextChunkWindow]:
    """Build fully-covering chunk windows using tokenizer overflow + offsets.

    The returned windows are character offsets into the original text. Windows
    are derived from tokenizer offsets (fast tokenizers only), with optional
    extra windows for uncovered gaps (typically whitespace) to ensure complete
    coverage of the original string.
    """

    if text is None:
        text = ""
    text = str(text)
    text_len = len(text)
    if text_len == 0:
        return [TextChunkWindow(text_start=0, text_end=0)]

    max_input_tokens = int(max_input_tokens)
    if max_input_tokens < 2:
        raise ValueError("max_input_tokens must be >= 2")
    if overlap_tokens is None:
        overlap_tokens = deterministic_overlap_tokens(max_input_tokens)
    overlap_tokens = max(0, min(int(overlap_tokens), max_input_tokens - 1))

    def _chunk_no_gaps(*, segment: str) -> list[TextChunkWindow]:
        _require_fast_tokenizer(tokenizer)
        enc = tokenizer(
            segment,
            add_special_tokens=False,
            truncation=True,
            max_length=max_input_tokens,
            stride=overlap_tokens,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mapping = getattr(enc, "get", lambda *_: None)("offset_mapping")
        if offset_mapping is None:
            offset_mapping = getattr(enc, "offset_mapping", None)
        chunks = _coerce_offset_chunks(offset_mapping)
        if not chunks:
            raise RuntimeError("tokenizer did not return offset mappings for chunking")

        out: list[TextChunkWindow] = []
        seg_len = len(segment)
        for chunk_offsets in chunks:
            valid = [(s, e) for s, e in chunk_offsets if isinstance(s, int) and isinstance(e, int) and e > s]
            if not valid:
                continue
            start = min(s for s, _ in valid)
            end = max(e for _, e in valid)
            start = max(0, min(int(start), seg_len))
            end = max(0, min(int(end), seg_len))
            if end > start:
                out.append(TextChunkWindow(text_start=start, text_end=end))
        return sorted(out, key=lambda w: (w.text_start, w.text_end))

    windows = _chunk_no_gaps(segment=text)
    if not windows:
        # No offsets for any content: allow pure-whitespace input, otherwise fail closed.
        if text.strip():
            raise RuntimeError("tokenizer produced no offsets for non-empty text (refuse to risk silent truncation)")
        return [TextChunkWindow(text_start=0, text_end=text_len)]

    # Ensure complete character coverage by adding windows for any gaps.
    covered: list[TextChunkWindow] = []
    cursor = 0
    for w in windows:
        if w.text_start > cursor:
            gap_start = cursor
            gap_end = w.text_start
            gap_text = text[gap_start:gap_end]
            gap_windows = _chunk_no_gaps(segment=gap_text)
            if not gap_windows:
                if gap_text.strip():
                    raise RuntimeError("tokenizer produced no offsets for gap content (refuse to risk silent truncation)")
                covered.append(TextChunkWindow(text_start=gap_start, text_end=gap_end))
            else:
                covered.extend(
                    [TextChunkWindow(text_start=gap_start + gw.text_start, text_end=gap_start + gw.text_end) for gw in gap_windows]
                )
        covered.append(w)
        cursor = max(cursor, w.text_end)
    if cursor < text_len:
        gap_start = cursor
        gap_end = text_len
        gap_text = text[gap_start:gap_end]
        gap_windows = _chunk_no_gaps(segment=gap_text)
        if not gap_windows:
            if gap_text.strip():
                raise RuntimeError("tokenizer produced no offsets for gap content (refuse to risk silent truncation)")
            covered.append(TextChunkWindow(text_start=gap_start, text_end=gap_end))
        else:
            covered.extend(
                [TextChunkWindow(text_start=gap_start + gw.text_start, text_end=gap_start + gw.text_end) for gw in gap_windows]
            )

    # Merge adjacent windows that are identical or empty.
    normalized: list[TextChunkWindow] = []
    for w in covered:
        if w.text_end <= w.text_start:
            continue
        if normalized and normalized[-1].text_end == w.text_start:
            # Keep separate windows (deterministic) unless the new window is a pure gap.
            pass
        normalized.append(w)
    return normalized


def _reasonable_max_length(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        value = int(value)
    except Exception:
        return None
    if value < 2:
        return None
    # Many tokenizers use huge sentinels for "infinite".
    if value > 100_000:
        return None
    return value


def effective_max_tokens_for_token_classifier(*, model: Any, tokenizer: Any) -> int:
    """Return the maximum number of *text tokens* (excluding special tokens)."""

    _require_fast_tokenizer(tokenizer)

    max_pos = _reasonable_max_length(getattr(getattr(model, "config", None), "max_position_embeddings", None))
    if max_pos is None:
        max_pos = _reasonable_max_length(getattr(tokenizer, "model_max_length", None))
    if max_pos is None:
        raise RuntimeError("unable to determine model context length for token-classifier")

    specials = int(getattr(tokenizer, "num_special_tokens_to_add", lambda **_: 0)(pair=False))
    cap = int(max_pos) - max(0, specials)
    if cap < 2:
        raise RuntimeError("invalid derived token-classifier context length")
    return cap

