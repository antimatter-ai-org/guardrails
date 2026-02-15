from __future__ import annotations

from app.core.analysis.span_normalizer import normalize_detections
from app.models.entities import Detection


def test_location_span_expands_to_address_chain_when_enabled() -> None:
    text = "Manzil: O'zbekiston, Toshkent shahri, Chilonzor tumani, 5-mavze, 12-uy."
    start = text.index("Toshkent")
    end = start + len("Toshkent shahri")
    detections = [
        Detection(
            start=start,
            end=end,
            text=text[start:end],
            label="LOCATION",
            score=0.9,
            detector="gliner",
            metadata={"canonical_label": "location"},
        )
    ]

    out = normalize_detections(
        text=text,
        detections=detections,
    )

    assert len(out) == 1
    assert not out[0].text.startswith("Manzil:")
    assert out[0].text.startswith("Toshkent")
    assert "12-uy" in out[0].text


def test_location_span_keeps_original_when_context_is_not_address_like() -> None:
    text = "Trip to Berlin was great yesterday."
    start = text.index("Berlin")
    end = start + len("Berlin")
    detections = [
        Detection(
            start=start,
            end=end,
            text=text[start:end],
            label="LOCATION",
            score=0.8,
            detector="gliner",
            metadata={"canonical_label": "location"},
        )
    ]

    out = normalize_detections(
        text=text,
        detections=detections,
    )

    assert len(out) == 1
    assert out[0].text == "Berlin"


def test_postprocess_disabled_keeps_spans_unchanged() -> None:
    text = "ID (AB1234567)"
    start = text.index("(")
    end = text.index(")") + 1
    detection = Detection(
        start=start,
        end=end,
        text=text[start:end],
        label="DOCUMENT_NUMBER",
        score=0.9,
        detector="regex",
        metadata={"canonical_label": "identifier"},
    )

    out = normalize_detections(text=text, detections=[detection])
    assert out[0].start >= detection.start
    assert out[0].end <= detection.end
    assert out[0].text == "AB1234567"
