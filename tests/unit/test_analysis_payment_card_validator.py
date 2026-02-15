from __future__ import annotations

from app.config import AnalysisConfig, AnalyzerProfile, PolicyConfig, PolicyDefinition, RecognizerDefinition
from app.core.analysis.service import PresidioAnalysisService


def _build_service() -> PresidioAnalysisService:
    config = PolicyConfig(
        default_policy="default",
        policies={
            "default": PolicyDefinition(
                mode="mask",
                analyzer_profile="profile",
                min_score=0.5,
                storage_ttl_seconds=300,
                placeholder_prefix="GR",
            )
        },
        analyzer_profiles={
            "profile": AnalyzerProfile(
                analysis=AnalysisConfig(
                    recognizers=["card_regex"],
                )
            )
        },
        recognizer_definitions={
            "card_regex": RecognizerDefinition(
                type="regex",
                enabled=True,
                params={
                    "patterns": [
                        {
                            "name": "bank_card",
                            "label": "CREDIT_CARD",
                            "pattern": r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)",
                            "score": 0.95,
                        }
                    ]
                },
            )
        },
    )
    return PresidioAnalysisService(config)


def test_payment_card_validator_accepts_luhn_number() -> None:
    service = _build_service()
    detections = service.analyze_text(
        text="Use card 4242 4242 4242 4242 for testing",
        profile_name="profile",
        policy_min_score=0.5,
    )
    assert len(detections) == 1
    assert detections[0].metadata.get("canonical_label") == "payment_card"


def test_payment_card_validator_non_luhn_without_context_is_kept_with_lower_score() -> None:
    service = _build_service()
    detections = service.analyze_text(
        text="Random number 1234567890123456 in text",
        profile_name="profile",
        policy_min_score=0.5,
    )
    assert len(detections) == 1
    assert detections[0].score < 0.95


def test_payment_card_validator_accepts_grouped_number_with_card_context() -> None:
    service = _build_service()
    detections = service.analyze_text(
        text="Номер карты: 1234 5678 9012 3456",
        profile_name="profile",
        policy_min_score=0.5,
    )
    assert len(detections) == 1
    assert detections[0].metadata.get("canonical_label") == "payment_card"


def test_payment_card_validator_accepts_grouped_number_without_context_for_recall() -> None:
    service = _build_service()
    detections = service.analyze_text(
        text="Random number 1234 5678 9012 3456 in text",
        profile_name="profile",
        policy_min_score=0.5,
    )
    assert len(detections) == 1
    assert detections[0].metadata.get("canonical_label") == "payment_card"


def test_payment_card_validator_rejects_uniform_digits() -> None:
    service = _build_service()
    detections = service.analyze_text(
        text="Номер карты 1111 1111 1111 1111",
        profile_name="profile",
        policy_min_score=0.5,
    )
    assert detections == []
