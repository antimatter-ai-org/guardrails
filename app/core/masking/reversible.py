from __future__ import annotations

from dataclasses import replace

from app.models.entities import Detection, MaskingResult, UnmaskingResult


class ReversibleMaskingEngine:
    def __init__(self, placeholder_prefix: str) -> None:
        self._placeholder_prefix = placeholder_prefix

    @staticmethod
    def resolve_overlaps(detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        priority_class = {
            "secret": 5,
            "identifier": 4,
            "ip": 4,
            "email": 3,
            "phone": 3,
            "payment_card": 3,
            "date": 2,
            "person": 1,
            "location": 1,
            "organization": 1,
        }

        def rank(item: Detection) -> tuple[float, int, int, int]:
            canonical = str(item.metadata.get("canonical_label") or "").lower()
            return (
                float(item.score),
                priority_class.get(canonical, 0),
                item.end - item.start,
                -item.start,
            )

        sorted_by_priority = sorted(detections, key=rank, reverse=True)
        selected: list[Detection] = []
        for candidate in sorted_by_priority:
            if any(not (candidate.end <= item.start or candidate.start >= item.end) for item in selected):
                continue
            selected.append(candidate)
        return sorted(selected, key=lambda item: item.start)

    def mask(self, text: str, detections: list[Detection]) -> MaskingResult:
        findings = self.resolve_overlaps([item for item in detections if item.end > item.start])
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

        remapped = [replace(item, text=text[item.start : item.end]) for item in findings]
        return MaskingResult(text=masked, placeholders=placeholders, detections=remapped)

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
