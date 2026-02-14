from __future__ import annotations

from app.config import PolicyDefinition
from app.eval.predictor import detect_only, merge_cascade_detections, sample_uncertainty_score
from app.models.entities import Detection


class _StaticDetector:
    def __init__(self, name: str, detections: list[Detection]) -> None:
        self.name = name
        self._detections = detections

    def detect(self, text: str) -> list[Detection]:
        _ = text
        return list(self._detections)


def test_detect_only_weighted_prefers_regex_ip_over_generic_model() -> None:
    text = "ip 193.51.208.14"
    regex_detection = Detection(
        start=3,
        end=16,
        text="193.51.208.14",
        label="IP_ADDRESS",
        score=0.72,
        detector="network_pii_regex",
    )
    gliner_detection = Detection(
        start=3,
        end=16,
        text="193.51.208.14",
        label="GLINER_phone number",
        score=0.84,
        detector="gliner_pii_multilingual",
    )

    findings = detect_only(
        text=text,
        policy=PolicyDefinition(mode="mask", detectors=[], min_score=0.5),
        detectors=[
            _StaticDetector("network_pii_regex", [regex_detection]),
            _StaticDetector("gliner_pii_multilingual", [gliner_detection]),
        ],
    )
    assert len(findings) == 1
    assert findings[0].label == "IP_ADDRESS"


def test_sample_uncertainty_score_returns_one_for_empty_stage_a() -> None:
    assert sample_uncertainty_score("text", stage_a_detections=[], min_score=0.55) == 1.0


def test_merge_cascade_detections_uses_weighted_overlap_resolution() -> None:
    stage_a = [
        Detection(
            start=0,
            end=10,
            text="12.03.2022",
            label="GLINER_person",
            score=0.9,
            detector="gliner_pii_multilingual",
        )
    ]
    stage_b = [
        Detection(
            start=0,
            end=10,
            text="12.03.2022",
            label="DATE",
            score=0.7,
            detector="date_pii_regex",
        )
    ]
    merged = merge_cascade_detections(stage_a, stage_b)
    assert len(merged) == 1
    assert merged[0].label == "DATE"
