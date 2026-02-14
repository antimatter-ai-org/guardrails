from __future__ import annotations

from app.detectors.base import Detector
from app.models.entities import Detection


class GlinerDetector(Detector):
    def __init__(
        self,
        name: str,
        model_name: str,
        labels: list[str],
        threshold: float = 0.5,
    ) -> None:
        super().__init__(name)
        try:
            from gliner import GLiNER
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("gliner is not installed. Install with guardrails-service[ml].") from exc

        self._model = GLiNER.from_pretrained(model_name)
        self._labels = labels
        self._threshold = threshold

    def detect(self, text: str) -> list[Detection]:
        raw_predictions = self._model.predict_entities(text, self._labels, threshold=self._threshold)
        findings: list[Detection] = []
        for pred in raw_predictions:
            findings.append(
                Detection(
                    start=int(pred["start"]),
                    end=int(pred["end"]),
                    text=pred["text"],
                    label=f"GLINER_{pred['label']}",
                    score=float(pred.get("score", self._threshold)),
                    detector=self.name,
                    metadata={"source": "gliner"},
                )
            )
        return findings
