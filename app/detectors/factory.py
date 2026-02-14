from __future__ import annotations

import logging
from collections.abc import Mapping

from app.config import DetectorDefinition
from app.detectors.base import Detector
from app.detectors.entropy_detector import EntropyDetector
from app.detectors.gliner_detector import GlinerDetector
from app.detectors.natasha_detector import NatashaDetector
from app.detectors.regex_detector import RegexDetector
from app.detectors.secret_detector import SecretRegexDetector
from app.settings import settings

logger = logging.getLogger(__name__)


class DetectorRegistry:
    def __init__(self, detectors: Mapping[str, Detector]) -> None:
        self._detectors = dict(detectors)

    def get(self, name: str) -> Detector | None:
        return self._detectors.get(name)

    def names(self) -> list[str]:
        return sorted(self._detectors)


def _build_detector(name: str, definition: DetectorDefinition) -> Detector:
    detector_type = definition.type.lower()
    params = definition.params

    if detector_type == "regex":
        return RegexDetector(name=name, patterns=params.get("patterns", []))
    if detector_type == "secret_regex":
        return SecretRegexDetector(name=name, patterns=params.get("patterns"))
    if detector_type == "entropy":
        return EntropyDetector(
            name=name,
            min_length=int(params.get("min_length", 20)),
            entropy_threshold=float(params.get("entropy_threshold", 3.6)),
            pattern=str(params.get("pattern", r"\b[A-Za-z0-9_\-/+=]{20,}\b")),
        )
    if detector_type == "natasha":
        return NatashaDetector(name=name, score=float(params.get("score", 0.7)))
    if detector_type == "gliner":
        return GlinerDetector(
            name=name,
            model_name=str(params.get("model_name", "urchade/gliner_multi-v2.1")),
            labels=[str(item) for item in params.get("labels", [])],
            threshold=float(params.get("threshold", 0.5)),
            backend=str(params.get("backend", settings.gliner_backend)),
            device=str(params.get("device", settings.gliner_device)),
            use_fp16_on_cuda=bool(params.get("use_fp16_on_cuda", settings.gliner_use_fp16_on_cuda)),
        )

    raise ValueError(f"Unsupported detector type: {detector_type}")


def build_registry(definitions: Mapping[str, DetectorDefinition]) -> DetectorRegistry:
    detectors: dict[str, Detector] = {}
    for name, definition in definitions.items():
        if not definition.enabled:
            logger.info("detector disabled: %s", name)
            continue
        try:
            detectors[name] = _build_detector(name, definition)
            logger.info("detector loaded: %s (%s)", name, definition.type)
        except Exception as exc:
            logger.warning("detector %s skipped due to initialization error: %s", name, exc)
    return DetectorRegistry(detectors)
