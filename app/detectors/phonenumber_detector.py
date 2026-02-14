from __future__ import annotations

from app.detectors.base import Detector
from app.models.entities import Detection


class PhoneNumberDetector(Detector):
    def __init__(
        self,
        name: str,
        score: float = 0.91,
        label: str = "PHONE_LIB",
        regions: list[str] | None = None,
        min_digits: int = 10,
    ) -> None:
        super().__init__(name)
        self._score = score
        self._label = label
        self._regions = regions or ["RU", "UA", "US", "GB", "DE", "PL", "KZ"]
        self._min_digits = max(6, int(min_digits))

        try:
            import phonenumbers
            from phonenumbers import Leniency, PhoneNumberMatcher, is_valid_number
        except Exception as exc:
            raise RuntimeError(
                "phonenumbers is not installed. Install project dependencies (for example: `uv sync`)."
            ) from exc

        self._phonenumbers = phonenumbers
        self._leniency = Leniency
        self._matcher_cls = PhoneNumberMatcher
        self._is_valid_number = is_valid_number

    def detect(self, text: str) -> list[Detection]:
        findings: list[Detection] = []
        seen: set[tuple[int, int]] = set()

        for region in self._regions:
            try:
                matcher = self._matcher_cls(text, region, leniency=self._leniency.VALID)
            except Exception:
                continue

            for match in matcher:
                start = int(match.start)
                end = int(match.end)
                if end <= start:
                    continue
                if (start, end) in seen:
                    continue

                raw = text[start:end]
                digit_count = sum(char.isdigit() for char in raw)
                if digit_count < self._min_digits:
                    continue

                try:
                    if not self._is_valid_number(match.number):
                        continue
                except Exception:
                    continue

                seen.add((start, end))
                findings.append(
                    Detection(
                        start=start,
                        end=end,
                        text=raw,
                        label=self._label,
                        score=self._score,
                        detector=self.name,
                        metadata={"source": "phonenumbers", "region": region},
                    )
                )

        return sorted(findings, key=lambda item: (item.start, item.end))
