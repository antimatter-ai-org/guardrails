from __future__ import annotations

from app.detectors.dateparser_detector import DateParserDetector
from app.detectors.entropy_detector import EntropyDetector
from app.detectors.ipaddress_detector import IPAddressDetector
from app.detectors.phonenumber_detector import PhoneNumberDetector
from app.detectors.regex_detector import RegexDetector


def test_regex_detector_finds_ru_phone_and_email() -> None:
    detector = RegexDetector(
        name="ru",
        patterns=[
                {
                    "name": "phone",
                    "label": "PHONE",
                    "pattern": r"(?:\+7|8)\d{10}",
                    "score": 0.9,
                },
                {
                    "name": "email",
                    "label": "EMAIL",
                    "pattern": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                    "score": 0.9,
                },
        ],
    )
    text = "Позвони мне +79991234567 и пиши на oleg@example.com"
    findings = detector.detect(text)

    labels = {item.label for item in findings}
    assert labels == {"PHONE", "EMAIL"}


def test_entropy_detector_finds_high_entropy_candidate() -> None:
    detector = EntropyDetector(name="entropy", min_length=20, entropy_threshold=3.5)
    text = "token=stripe_example_key_value_0002"
    findings = detector.detect(text)
    assert findings
    assert findings[0].label == "SECRET_HIGH_ENTROPY"


def test_phonenumber_detector_detects_valid_phone() -> None:
    detector = PhoneNumberDetector(name="phone", regions=["UA", "RU", "US"])
    text = "Контакты: +380661998877 и backup +1 (415) 555-0123"
    findings = detector.detect(text)
    assert len(findings) >= 2
    assert all(item.label == "PHONE_LIB" for item in findings)


def test_dateparser_detector_detects_textual_ru_and_en_dates() -> None:
    detector = DateParserDetector(name="date", languages=["ru", "en"])
    text = "Срок: 22 мая 1999 года, then updated on 14 February 2024."
    findings = detector.detect(text)
    assert len(findings) >= 2
    assert all(item.label == "DATE_TEXT" for item in findings)


def test_ipaddress_detector_detects_ipv6_mixed_and_cidr() -> None:
    detector = IPAddressDetector(name="ip")
    text = "Route 2001:DB8::FFFF:10.10.2.1 and net 10.12.0.0/16 should be masked."
    findings = detector.detect(text)
    values = {item.text for item in findings}
    assert "2001:DB8::FFFF:10.10.2.1" in values
    assert "10.12.0.0/16" in values
