from __future__ import annotations

from app.runtime.gliner_word_chunking import chunk_text_by_gliner_words, split_gliner_words


def _assert_full_char_coverage(text: str, windows: list[object]) -> None:
    n = len(text)
    covered = [False] * (n + 1)
    for w in windows:
        start = int(getattr(w, "text_start"))
        end = int(getattr(w, "text_end"))
        assert 0 <= start <= end <= n
        for i in range(start, end):
            covered[i] = True
    # Full coverage: every character index must be included in at least one window.
    for i in range(n):
        assert covered[i], f"gap at char index {i}"


def test_split_gliner_words_matches_punct_and_words() -> None:
    text = "  Hello, world!\nEmail: a@b.com  "
    tokens = split_gliner_words(text)
    assert tokens
    # Leading/trailing whitespace is not tokenized but must be covered by chunking.
    assert tokens[0].token == "Hello"
    assert any(t.token == "," for t in tokens)
    assert any(t.token == "!" for t in tokens)


def test_chunking_covers_all_characters_including_whitespace() -> None:
    text = "  A  B\n\nC   D\tE  "
    windows = chunk_text_by_gliner_words(text=text, max_text_words=2, overlap_words=1)
    assert windows
    assert int(windows[0].text_start) == 0
    assert int(windows[-1].text_end) == len(text)
    _assert_full_char_coverage(text, windows)


def test_chunking_is_deterministic() -> None:
    text = "X Y Z W V U T"
    a = chunk_text_by_gliner_words(text=text, max_text_words=3, overlap_words=1)
    b = chunk_text_by_gliner_words(text=text, max_text_words=3, overlap_words=1)
    assert a == b


def test_chunking_handles_whitespace_only_text() -> None:
    text = " \n\t  "
    windows = chunk_text_by_gliner_words(text=text, max_text_words=3, overlap_words=1)
    assert windows == [type(windows[0])(text_start=0, text_end=len(text))]

