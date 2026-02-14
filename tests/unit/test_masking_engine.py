from __future__ import annotations

from app.detectors.base import Detector
from app.masking.engine import SensitiveDataMasker
from app.models.entities import Detection


class StaticDetector(Detector):
    def __init__(self, name: str, findings: list[Detection]) -> None:
        super().__init__(name)
        self._findings = findings

    def detect(self, text: str) -> list[Detection]:
        return self._findings


def test_mask_and_unmask_with_overlap_resolution() -> None:
    text = "client secret is 1234-5678 and email is ivan@example.com"
    detections = [
        Detection(start=17, end=26, text="1234-5678", label="TOKEN", score=0.9, detector="test"),
        Detection(start=17, end=31, text="1234-5678 and ", label="NOISE", score=0.3, detector="test"),
        Detection(start=40, end=56, text="ivan@example.com", label="EMAIL", score=0.95, detector="test"),
    ]
    masker = SensitiveDataMasker(
        detectors=[StaticDetector("static", detections)],
        min_score=0.5,
        placeholder_prefix="GR",
    )

    masked = masker.mask(text)

    assert "1234-5678" not in masked.text
    assert "ivan@example.com" not in masked.text
    assert len(masked.placeholders) == 2

    unmasked = SensitiveDataMasker.unmask(masked.text, masked.placeholders)
    assert unmasked.text == text
