from __future__ import annotations

from app.detectors.base import Detector
from app.models.entities import Detection
from app.runtime.gliner_runtime import build_gliner_runtime


class GlinerDetector(Detector):
    def __init__(
        self,
        name: str,
        model_name: str,
        labels: list[str],
        threshold: float = 0.5,
        backend: str = "local_torch",
        device: str = "auto",
        use_fp16_on_cuda: bool = False,
    ) -> None:
        super().__init__(name)
        self._runtime = build_gliner_runtime(
            backend=backend,
            model_name=model_name,
            preferred_device=device,
            use_fp16_on_cuda=use_fp16_on_cuda,
        )
        self._labels = labels
        self._threshold = threshold
        self._backend = backend
        self._device = getattr(self._runtime, "device", device)

    def detect(self, text: str) -> list[Detection]:
        raw_predictions = self._runtime.predict_entities(text, self._labels, threshold=self._threshold)
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
                    metadata={"source": "gliner", "backend": self._backend, "device": self._device},
                )
            )
        return findings
