from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

_WORD_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S", re.UNICODE)
_SENTENCE_END_CHARS = {".", "!", "?", "。", "！", "？", ";", "\n"}


@dataclass(slots=True, frozen=True)
class GlinerChunkingConfig:
    enabled: bool = True
    max_tokens: int = 320
    overlap_tokens: int = 64
    max_chunks: int = 64
    boundary_lookback_tokens: int = 24

    def normalized(self) -> GlinerChunkingConfig:
        max_tokens = max(2, int(self.max_tokens))
        overlap_tokens = max(0, min(int(self.overlap_tokens), max_tokens - 1))
        max_chunks = max(1, int(self.max_chunks))
        boundary_lookback_tokens = max(0, int(self.boundary_lookback_tokens))
        return GlinerChunkingConfig(
            enabled=bool(self.enabled),
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            max_chunks=max_chunks,
            boundary_lookback_tokens=boundary_lookback_tokens,
        )


@dataclass(slots=True, frozen=True)
class TextChunkWindow:
    text_start: int
    text_end: int
    token_start: int
    token_end: int


def _word_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _WORD_PATTERN.finditer(text)]


def _snap_chunk_end(
    text: str,
    spans: list[tuple[int, int]],
    token_start: int,
    raw_end_token: int,
    lookback_tokens: int,
) -> int:
    if raw_end_token >= len(spans) or lookback_tokens <= 0:
        return raw_end_token

    lower = max(token_start + 1, raw_end_token - lookback_tokens)
    for idx in range(raw_end_token - 1, lower - 1, -1):
        end_char_idx = spans[idx][1] - 1
        if end_char_idx < 0 or end_char_idx >= len(text):
            continue
        if text[end_char_idx] in _SENTENCE_END_CHARS:
            snapped = idx + 1
            # Avoid over-fragmenting text into very short chunks.
            if snapped - token_start >= max(24, (raw_end_token - token_start) // 3):
                return snapped
            break
    return raw_end_token


def build_chunk_windows(text: str, config: GlinerChunkingConfig) -> list[TextChunkWindow]:
    cfg = config.normalized()
    spans = _word_spans(text)

    if not spans:
        return [TextChunkWindow(text_start=0, text_end=len(text), token_start=0, token_end=0)]
    if not cfg.enabled or len(spans) <= cfg.max_tokens:
        return [TextChunkWindow(text_start=0, text_end=len(text), token_start=0, token_end=len(spans))]

    windows: list[TextChunkWindow] = []
    stride = cfg.max_tokens - cfg.overlap_tokens
    token_start = 0
    total_tokens = len(spans)

    while token_start < total_tokens and len(windows) < cfg.max_chunks:
        raw_end = min(token_start + cfg.max_tokens, total_tokens)
        token_end = _snap_chunk_end(
            text=text,
            spans=spans,
            token_start=token_start,
            raw_end_token=raw_end,
            lookback_tokens=cfg.boundary_lookback_tokens,
        )
        if token_end <= token_start:
            token_end = raw_end

        text_start = spans[token_start][0]
        text_end = spans[token_end - 1][1]
        windows.append(
            TextChunkWindow(
                text_start=text_start,
                text_end=text_end,
                token_start=token_start,
                token_end=token_end,
            )
        )
        if token_end >= total_tokens:
            break

        next_start = max(0, token_end - cfg.overlap_tokens)
        if next_start <= token_start:
            next_start = token_start + max(1, stride)
        token_start = next_start

    # Ensure tail coverage when capped by max_chunks.
    if windows and windows[-1].token_end < total_tokens:
        tail_end = total_tokens
        tail_start = max(0, tail_end - cfg.max_tokens)
        tail_window = TextChunkWindow(
            text_start=spans[tail_start][0],
            text_end=spans[tail_end - 1][1],
            token_start=tail_start,
            token_end=tail_end,
        )
        if windows[-1] != tail_window:
            if len(windows) >= cfg.max_chunks:
                windows[-1] = tail_window
            else:
                windows.append(tail_window)

    return windows


def run_chunked_inference(
    *,
    text: str,
    labels: list[str],
    threshold: float,
    chunking: GlinerChunkingConfig,
    predict_batch: Callable[[list[str], list[str], float], list[list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    windows = build_chunk_windows(text, chunking)
    chunk_texts = [text[window.text_start : window.text_end] for window in windows]
    chunk_predictions = predict_batch(chunk_texts, labels, threshold)

    dedup: dict[tuple[int, int, str], dict[str, Any]] = {}
    text_len = len(text)
    for window, predictions in zip(windows, chunk_predictions, strict=False):
        chunk_len = max(0, window.text_end - window.text_start)
        for item in predictions:
            if not isinstance(item, dict):
                continue

            local_start = int(item.get("start", -1))
            local_end = int(item.get("end", -1))
            if local_start < 0 or local_end <= local_start:
                continue

            local_start = max(0, min(local_start, chunk_len))
            local_end = max(0, min(local_end, chunk_len))
            if local_end <= local_start:
                continue

            global_start = window.text_start + local_start
            global_end = window.text_start + local_end
            if global_start < 0 or global_end > text_len or global_end <= global_start:
                continue

            label = str(item.get("label", ""))
            if not label:
                continue

            score = float(item.get("score", threshold))
            key = (global_start, global_end, label)
            existing = dedup.get(key)
            if existing is None or score > float(existing.get("score", 0.0)):
                dedup[key] = {
                    "start": global_start,
                    "end": global_end,
                    "text": text[global_start:global_end],
                    "label": label,
                    "score": score,
                }

    return sorted(dedup.values(), key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))
