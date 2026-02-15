from __future__ import annotations

from app.config import AnalysisConfig, AnalyzerProfile, PolicyConfig, PolicyDefinition, RecognizerDefinition
from app.core.analysis.service import PresidioAnalysisService


def _make_service() -> PresidioAnalysisService:
    config = PolicyConfig(
        default_policy="p",
        policies={
            "p": PolicyDefinition(
                mode="mask",
                analyzer_profile="profile",
                min_score=0.5,
                storage_ttl_seconds=300,
                placeholder_prefix="GR",
            )
        },
        analyzer_profiles={
            "profile": AnalyzerProfile(
                analysis=AnalysisConfig(recognizers=["bulk_regex"]),
            )
        },
        recognizer_definitions={
            "bulk_regex": RecognizerDefinition(
                type="regex",
                enabled=True,
                params={
                    "patterns": [
                        {
                            "name": "bulk",
                            "label": "DOCUMENT_NUMBER",
                            "pattern": r"\bX\d+\b",
                            "score": 0.95,
                        }
                    ]
                },
            )
        },
    )
    return PresidioAnalysisService(config)


def test_analysis_diagnostics_include_detector_stats_and_limits() -> None:
    service = _make_service()
    text = " ".join(f"X{idx}" for idx in range(400))

    detections, diagnostics = service.analyze_text_with_diagnostics(
        text=text,
        profile_name="profile",
        policy_min_score=0.5,
    )

    assert len(detections) == service._MAX_SAMPLE_DETECTIONS  # noqa: SLF001
    assert diagnostics.limit_flags["max_spans_truncated"] is True
    assert diagnostics.detector_timing_ms
    assert diagnostics.detector_span_counts
    assert diagnostics.elapsed_ms >= 0
    assert diagnostics.postprocess_mutations["overlaps_dropped"] >= 0


def test_analysis_timeout_flag_is_set_when_budget_is_exceeded() -> None:
    service = _make_service()
    service._MAX_SAMPLE_ANALYSIS_SECONDS = 0.0  # noqa: SLF001

    _, diagnostics = service.analyze_text_with_diagnostics(
        text="X1 X2 X3",
        profile_name="profile",
        policy_min_score=0.5,
    )

    assert diagnostics.limit_flags["analysis_timeout_exceeded"] is True
