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
        runtime_mode: str = "cpu",
        cpu_device: str = "auto",
        pytriton_url: str = "pytriton:8000",
        pytriton_model_name: str = "gliner",
        pytriton_init_timeout_s: float = 20.0,
        pytriton_infer_timeout_s: float = 30.0,
    ) -> None:
        super().__init__(name)
        self._runtime = build_gliner_runtime(
            runtime_mode=runtime_mode,
            model_name=model_name,
            cpu_device=cpu_device,
            pytriton_url=pytriton_url,
            pytriton_model_name=pytriton_model_name,
            pytriton_init_timeout_s=pytriton_init_timeout_s,
            pytriton_infer_timeout_s=pytriton_infer_timeout_s,
        )
        self._labels = labels
        self._threshold = threshold
        self._runtime_mode = runtime_mode
        self._device = getattr(self._runtime, "device", "unknown")

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
                    metadata={"source": "gliner", "runtime_mode": self._runtime_mode, "device": self._device},
                )
            )
        return findings
