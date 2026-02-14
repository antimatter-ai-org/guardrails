from __future__ import annotations

from dataclasses import replace

from app.detectors.base import Detector
from app.models.entities import Detection, MaskingResult, UnmaskingResult


class SensitiveDataMasker:
    def __init__(
        self,
        detectors: list[Detector],
        min_score: float,
        placeholder_prefix: str,
    ) -> None:
        self._detectors = detectors
        self._min_score = min_score
        self._placeholder_prefix = placeholder_prefix

    def _collect(self, text: str) -> list[Detection]:
        findings: list[Detection] = []
        for detector in self._detectors:
            findings.extend(detector.detect(text))
        return [item for item in findings if item.end > item.start and item.score >= self._min_score]

    @staticmethod
    def _resolve_overlaps(detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        sorted_by_priority = sorted(
            detections,
            key=lambda item: (-item.score, -(item.end - item.start), item.start),
        )

        selected: list[Detection] = []
        for candidate in sorted_by_priority:
            if any(not (candidate.end <= existing.start or candidate.start >= existing.end) for existing in selected):
                continue
            selected.append(candidate)

        return sorted(selected, key=lambda item: item.start)

    def mask(self, text: str) -> MaskingResult:
        findings = self._resolve_overlaps(self._collect(text))
        if not findings:
            return MaskingResult(text=text, placeholders={}, detections=[])

        placeholders: dict[str, str] = {}
        placeholder_by_original: dict[str, str] = {}

        masked = text
        for finding in reversed(findings):
            original = masked[finding.start : finding.end]
            placeholder = placeholder_by_original.get(original)
            if not placeholder:
                placeholder = f"<{self._placeholder_prefix}:{len(placeholders) + 1:04d}>"
                placeholders[placeholder] = original
                placeholder_by_original[original] = placeholder

            masked = masked[: finding.start] + placeholder + masked[finding.end :]

        remapped_findings: list[Detection] = []
        for finding in findings:
            remapped_findings.append(replace(finding, text=text[finding.start : finding.end]))

        return MaskingResult(text=masked, placeholders=placeholders, detections=remapped_findings)

    @staticmethod
    def unmask(text: str, placeholders: dict[str, str]) -> UnmaskingResult:
        if not placeholders:
            return UnmaskingResult(text=text, replaced=0)

        restored = text
        replaced = 0
        for placeholder, original in sorted(placeholders.items(), key=lambda item: len(item[0]), reverse=True):
            count = restored.count(placeholder)
            if count:
                restored = restored.replace(placeholder, original)
                replaced += count

        return UnmaskingResult(text=restored, replaced=replaced)
