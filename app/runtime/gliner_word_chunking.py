from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.runtime.tokenizer_chunking import TextChunkWindow


# Match GLiNER's default WhitespaceTokenSplitter:
# gliner/data_processing/tokenizer.py::WhitespaceTokenSplitter
_WHITESPACE_SPLITTER_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")


@dataclass(frozen=True, slots=True)
class WordToken:
    token: str
    start: int
    end: int


def split_gliner_words(text: str) -> list[WordToken]:
    """Split text into GLiNER "word" tokens with char offsets (default splitter)."""
    out: list[WordToken] = []
    for match in _WHITESPACE_SPLITTER_RE.finditer(text or ""):
        out.append(WordToken(token=match.group(0), start=match.start(), end=match.end()))
    return out


def deterministic_overlap_words(*, max_len_words: int, max_width: int) -> int:
    # Deterministic, non-configurable overlap. Include max_width to keep entity spans stable.
    base = max(16, int(max_len_words) // 8)
    value = max(base, int(max_width))
    return max(0, min(value, max(0, int(max_len_words) - 1)))


def chunk_text_by_gliner_words(
    *,
    text: str,
    max_text_words: int,
    overlap_words: int,
) -> list[TextChunkWindow]:
    """Chunk text into fully-covering char windows based on GLiNER word splitter.

    Windows are defined over token indices but expanded to char spans that also
    include inter-token whitespace (by ending at the next token start).
    """
    if max_text_words < 1:
        raise ValueError("max_text_words must be >= 1")
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")

    tokens = split_gliner_words(text)
    n = len(tokens)
    if n == 0:
        # Still fully cover text, even if it is only whitespace.
        if text:
            return [TextChunkWindow(text_start=0, text_end=len(text))]
        return [TextChunkWindow(text_start=0, text_end=0)]

    step = max(1, max_text_words - overlap_words)
    windows: list[TextChunkWindow] = []
    idx = 0
    while idx < n:
        token_start = idx
        token_end = min(n, idx + max_text_words)

        # Char span that covers tokens plus trailing whitespace up to the next token.
        if token_start == 0:
            text_start = 0
        else:
            text_start = tokens[token_start].start

        if token_end >= n:
            text_end = len(text)
        else:
            text_end = tokens[token_end].start

        if text_end <= text_start:
            # Defensive: should not happen, but never emit empty windows.
            idx += step
            continue

        windows.append(
            TextChunkWindow(
                text_start=text_start,
                text_end=text_end,
            )
        )
        idx += step

    # Ensure full coverage of the tail, even if the last token ended early.
    if windows and windows[-1].text_end < len(text):
        last = windows[-1]
        windows[-1] = TextChunkWindow(
            text_start=last.text_start,
            text_end=len(text),
        )

    return windows


def gliner_prompt_len_words(labels: list[str]) -> int:
    # GLiNER prompt format is: [ENT] label per label, plus [SEP].
    uniq = list(dict.fromkeys([str(l) for l in labels if str(l).strip()]))
    return 2 * len(uniq) + 1


def build_prompt_tokens_for_length_check(*, ent_token: str, sep_token: str, labels: list[str]) -> list[str]:
    uniq = list(dict.fromkeys([str(l) for l in labels if str(l).strip()]))
    prompt: list[str] = []
    for lab in uniq:
        prompt.append(ent_token)
        prompt.append(lab)
    prompt.append(sep_token)
    return prompt
