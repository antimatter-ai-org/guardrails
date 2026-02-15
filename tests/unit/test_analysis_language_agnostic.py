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
                analysis=AnalysisConfig(recognizers=["ru_phone_regex", "en_ssn_regex"]),
            )
        },
        recognizer_definitions={
            "ru_phone_regex": RecognizerDefinition(
                type="regex",
                enabled=True,
                params={
                    "patterns": [
                        {
                            "name": "ru_phone",
                            "label": "PHONE_NUMBER",
                            "pattern": r"(?<!\d)(?:\+7|8)\d{10}(?!\d)",
                            "score": 0.95,
                        }
                    ]
                },
            ),
            "en_ssn_regex": RecognizerDefinition(
                type="regex",
                enabled=True,
                params={
                    "patterns": [
                        {
                            "name": "us_ssn",
                            "label": "US_SSN",
                            "pattern": r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
                            "score": 0.99,
                        }
                    ]
                },
            ),
        },
    )
    return PresidioAnalysisService(config)


def test_mixed_ru_en_text_is_detected_in_single_pass() -> None:
    service = _build_service()
    detections = service.analyze_text(
        text="Телефон +79991234567 and SSN 123-45-6789",
        profile_name="profile",
        policy_min_score=0.5,
    )

    canonical = {item.metadata.get("canonical_label") for item in detections}
    assert "phone" in canonical
    assert "identifier" in canonical
    assert all("language" not in item.metadata for item in detections)


def test_en_regex_works_without_language_scoping() -> None:
    service = _build_service()
    detections = service.analyze_text(
        text="Confidential SSN: 123-45-6789",
        profile_name="profile",
        policy_min_score=0.5,
    )

    assert any(item.metadata.get("canonical_label") == "identifier" for item in detections)
