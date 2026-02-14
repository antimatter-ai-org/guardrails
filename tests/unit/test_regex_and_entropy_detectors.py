from __future__ import annotations

from app.detectors.entropy_detector import EntropyDetector
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
