from __future__ import annotations

from app.core.masking.reversible import ReversibleMaskingEngine
from app.models.entities import Detection


def test_mask_union_merge_prevents_leak_on_overlaps() -> None:
    text = "aa SECRET1 bb"
    s = text.index("SECRET1")
    e = s + len("SECRET1")
    detections = [
        Detection(
            start=s,
            end=e,
            text=text[s:e],
            label="SECRET",
            score=0.9,
            detector="det-a",
            metadata={"canonical_label": "secret"},
        ),
        # Overlapping partial span: should be union-merged, not dropped.
        Detection(
            start=s + 2,
            end=e - 1,
            text=text[s + 2 : e - 1],
            label="SECRET",
            score=0.8,
            detector="det-b",
            metadata={"canonical_label": "secret"},
        ),
    ]
    masker = ReversibleMaskingEngine("GR")
    masked = masker.mask(text, detections)
    assert "SECRET1" not in masked.text
    assert len(masked.detections) == 1
    assert masked.detections[0].start == s
    assert masked.detections[0].end == e


def test_mask_union_merge_keeps_separate_spans_separate() -> None:
    text = "xx SECRET1 yy SECRET2 zz"
    s1 = text.index("SECRET1")
    e1 = s1 + len("SECRET1")
    s2 = text.index("SECRET2")
    e2 = s2 + len("SECRET2")
    detections = [
        Detection(
            start=s1,
            end=e1,
            text=text[s1:e1],
            label="SECRET",
            score=0.9,
            detector="det",
            metadata={"canonical_label": "secret"},
        ),
        Detection(
            start=s2,
            end=e2,
            text=text[s2:e2],
            label="SECRET",
            score=0.95,
            detector="det",
            metadata={"canonical_label": "secret"},
        ),
    ]
    masker = ReversibleMaskingEngine("GR")
    masked = masker.mask(text, detections)
    assert "SECRET1" not in masked.text
    assert "SECRET2" not in masked.text
    assert len(masked.detections) == 2

