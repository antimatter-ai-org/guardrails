from __future__ import annotations

from app.eval.datasets.rubai import _build_spans_from_token_types, _select_token_types, _token_offsets
from app.eval.labels import canonicalize_rubai_gold_label


def test_token_offsets_align_with_whitespace_tokens() -> None:
    text = "Alice lives in Tashkent"
    offsets = _token_offsets(text)
    tokens = [text[start:end] for start, end in offsets]
    assert tokens == ["Alice", "lives", "in", "Tashkent"]


def test_build_spans_from_token_types_merges_adjacent_labels() -> None:
    text = "Alice Smith called from New York 12-12-2020"
    token_types = ["NAME", "NAME", "TEXT", "TEXT", "ADDRESS", "ADDRESS", "DATE"]

    spans = _build_spans_from_token_types(text, token_types)
    assert len(spans) == 3

    assert spans[0].label == "NAME"
    assert text[spans[0].start : spans[0].end] == "Alice Smith"
    assert spans[0].canonical_label == "person"

    assert spans[1].label == "ADDRESS"
    assert text[spans[1].start : spans[1].end] == "New York"
    assert spans[1].canonical_label == "location"

    assert spans[2].label == "DATE"
    assert text[spans[2].start : spans[2].end] == "12-12-2020"
    assert spans[2].canonical_label == "date"


def test_select_token_types_handles_placeholder_text_only_rows() -> None:
    text = "a b c d"
    token_types = _select_token_types(
        text=text,
        row_types=["TEXT"],
        denorm_types=[],
        labels=["TEXT"],
    )
    assert token_types == ["TEXT", "TEXT", "TEXT", "TEXT"]


def test_canonicalize_rubai_includes_payment_card() -> None:
    assert canonicalize_rubai_gold_label("CARD_NUMBER") == "payment_card"
