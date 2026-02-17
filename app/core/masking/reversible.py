from __future__ import annotations

from dataclasses import replace

from app.models.entities import Detection, MaskingResult, UnmaskingResult


class ReversibleMaskingEngine:
    def __init__(self, placeholder_prefix: str) -> None:
        self._placeholder_prefix = placeholder_prefix

    @staticmethod
    def union_merge_spans(detections: list[Detection]) -> list[Detection]:
        """Merge overlapping/adjacent spans into non-overlapping mask spans.

        This is leak-safe: we never drop coverage, only expand by union.
        """

        items = [d for d in detections if int(d.end) > int(d.start)]
        if not items:
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

        def rank(item: Detection) -> tuple[int, float, int, int]:
            canonical = str(item.metadata.get("canonical_label") or "").lower()
            return (
                priority_class.get(canonical, 0),
                float(item.score),
                int(item.end - item.start),
                -int(item.start),
            )

        items.sort(key=lambda d: (int(d.start), int(d.end)))
        merged: list[Detection] = []

        group: list[Detection] = []
        group_start = -1
        group_end = -1

        def flush() -> None:
            nonlocal group, group_start, group_end
            if not group:
                return
            best = max(group, key=rank)
            score = max(float(x.score) for x in group)
            detectors = sorted({str(x.detector or "") for x in group if str(x.detector or "").strip()})
            meta = dict(best.metadata or {})
            meta["merged_from_detectors"] = detectors
            merged.append(
                Detection(
                    start=int(group_start),
                    end=int(group_end),
                    text="",
                    label=str(best.label),
                    score=score,
                    detector=str(best.detector),
                    metadata=meta,
                )
            )
            group = []
            group_start = -1
            group_end = -1

        for item in items:
            start = int(item.start)
            end = int(item.end)
            if not group:
                group = [item]
                group_start = start
                group_end = end
                continue
            if start <= group_end:
                group.append(item)
                group_start = min(group_start, start)
                group_end = max(group_end, end)
                continue
            flush()
            group = [item]
            group_start = start
            group_end = end
        flush()
        return merged

    def mask(self, text: str, detections: list[Detection]) -> MaskingResult:
        findings = self.union_merge_spans(detections)
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
