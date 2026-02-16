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


def test_overlap_phone_wins_over_nested_identifier_when_phone_like() -> None:
    text = "Позвони на номер +7 9172283263 как можно скорее."
    phone_start = text.index("+7")
    phone_end = phone_start + len("+7 9172283263")
    id_start = text.index("9172283263")
    id_end = id_start + len("9172283263")
    detections = [
        Detection(
            start=phone_start,
            end=phone_end,
            text=text[phone_start:phone_end],
            label="PHONE",
            score=0.94,
            detector="ru_pii_regex:PHONE_NUMBER",
            metadata={"canonical_label": "phone", "entity_type": "PHONE_NUMBER"},
        ),
        Detection(
            start=id_start,
            end=id_end,
            text=text[id_start:id_end],
            label="IDENTIFIER",
            score=0.97,
            detector="ru_pii_regex:DOCUMENT_NUMBER",
            metadata={"canonical_label": "identifier", "entity_type": "DOCUMENT_NUMBER"},
        ),
    ]

    out = normalize_detections(text=text, detections=detections)
    assert len(out) == 1
    assert out[0].metadata["canonical_label"] == "phone"
    assert out[0].text == "+7 9172283263"


def test_overlap_ip_wins_over_conflicting_numeric_span() -> None:
    text = "Хост 192.168.1.34 должен быть доступен."
    ip_start = text.index("192.168.1.34")
    ip_end = ip_start + len("192.168.1.34")
    detections = [
        Detection(
            start=ip_start,
            end=ip_end,
            text=text[ip_start:ip_end],
            label="IP",
            score=0.80,
            detector="network_pii_regex:IP_ADDRESS",
            metadata={"canonical_label": "ip", "entity_type": "IP_ADDRESS"},
        ),
        Detection(
            start=ip_start,
            end=ip_end,
            text=text[ip_start:ip_end],
            label="PHONE",
            score=0.91,
            detector="gliner_pii_multilingual",
            metadata={"canonical_label": "phone", "entity_type": "PHONE_NUMBER"},
        ),
    ]

    out = normalize_detections(text=text, detections=detections)
    assert len(out) == 1
    assert out[0].metadata["canonical_label"] == "ip"


def test_overlap_payment_card_kept_when_luhn_signal_present() -> None:
    text = "Карта 5200828282828210 была подтверждена."
    start = text.index("5200828282828210")
    end = start + len("5200828282828210")
    detections = [
        Detection(
            start=start,
            end=end,
            text=text[start:end],
            label="PAYMENT_CARD",
            score=0.83,
            detector="ru_pii_regex:CREDIT_CARD",
            metadata={
                "canonical_label": "payment_card",
                "entity_type": "CREDIT_CARD",
                "payment_card_signal": {"luhn_valid": True, "grouped_like_card": False, "has_context": True},
            },
        ),
        Detection(
            start=start,
            end=end,
            text=text[start:end],
            label="IDENTIFIER",
            score=0.95,
            detector="ru_pii_regex:DOCUMENT_NUMBER",
            metadata={"canonical_label": "identifier", "entity_type": "DOCUMENT_NUMBER"},
        ),
    ]

    out = normalize_detections(text=text, detections=detections)
    assert len(out) == 1
    assert out[0].metadata["canonical_label"] == "payment_card"
