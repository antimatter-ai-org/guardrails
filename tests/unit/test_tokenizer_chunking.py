from __future__ import annotations

from dataclasses import asdict

from app.runtime.tokenizer_chunking import chunk_text


class _CharTokenizer:
    """Fake fast tokenizer that treats every character as a token.

    It supports overflowing windows via max_length + stride and returns offset mappings.
    """

    is_fast = True

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
        stride: int,
        return_overflowing_tokens: bool,
        return_offsets_mapping: bool,
    ):
        assert add_special_tokens is False
        assert truncation is True
        assert return_overflowing_tokens is True
        assert return_offsets_mapping is True
        tokens = [(idx, idx + 1) for idx in range(len(text))]
        if len(tokens) <= max_length:
            return {"offset_mapping": [tokens]}

        windows: list[list[tuple[int, int]]] = []
        token_start = 0
        while token_start < len(tokens):
            token_end = min(token_start + max_length, len(tokens))
            windows.append(tokens[token_start:token_end])
            if token_end >= len(tokens):
                break
            next_start = max(0, token_end - stride)
            if next_start <= token_start:
                next_start = token_start + 1
            token_start = next_start
        return {"offset_mapping": windows}


class _SkipPrefixTokenizer(_CharTokenizer):
    """Fake tokenizer that skips the first N characters (no offsets for them)."""

    def __init__(self, skip: int) -> None:
        self._skip = int(skip)

    def __call__(self, text: str, **kwargs):
        data = super().__call__(text[self._skip :], **kwargs)
        # Shift offsets forward into the original string.
        shifted: list[list[tuple[int, int]]] = []
        for chunk in data["offset_mapping"]:
            shifted.append([(s + self._skip, e + self._skip) for s, e in chunk])
        return {"offset_mapping": shifted}


def test_chunk_text_is_deterministic_and_covering() -> None:
    tokenizer = _CharTokenizer()
    text = "x" * 101
    windows = chunk_text(text=text, tokenizer=tokenizer, max_input_tokens=16, overlap_tokens=4)
    windows2 = chunk_text(text=text, tokenizer=tokenizer, max_input_tokens=16, overlap_tokens=4)
    assert [asdict(w) for w in windows] == [asdict(w) for w in windows2]

    covered = [False] * len(text)
    for w in windows:
        for idx in range(w.text_start, w.text_end):
            covered[idx] = True
    assert all(covered)


def test_chunk_text_overlaps_when_stride_is_set() -> None:
    tokenizer = _CharTokenizer()
    text = "abcdefghijklmnopqrstuvwxyz"
    windows = chunk_text(text=text, tokenizer=tokenizer, max_input_tokens=8, overlap_tokens=3)
    assert len(windows) > 1
    for left, right in zip(windows, windows[1:], strict=False):
        assert right.text_start < left.text_end


def test_chunk_text_adds_gap_windows_if_offsets_skip_prefix() -> None:
    tokenizer = _SkipPrefixTokenizer(skip=10)
    text = " " * 10 + "abcdef"
    windows = chunk_text(text=text, tokenizer=tokenizer, max_input_tokens=8, overlap_tokens=2)
    # Ensure the skipped prefix is still covered by at least one window.
    assert any(w.text_start == 0 and w.text_end == 10 for w in windows)

