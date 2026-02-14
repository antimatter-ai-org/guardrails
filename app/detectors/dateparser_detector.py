from __future__ import annotations

from app.detectors.base import Detector
from app.models.entities import Detection


class DateParserDetector(Detector):
    def __init__(
        self,
        name: str,
        score: float = 0.89,
        label: str = "DATE_TEXT",
        languages: list[str] | None = None,
        min_chars: int = 4,
        max_chars: int = 48,
    ) -> None:
        super().__init__(name)
        self._score = float(score)
        self._label = label
        self._languages = languages or ["ru", "uk", "en"]
        self._min_chars = max(1, int(min_chars))
        self._max_chars = max(self._min_chars, int(max_chars))

        try:
            from dateparser.search import search_dates
        except Exception as exc:
            raise RuntimeError(
                "dateparser is not installed. Install project dependencies (for example: `uv sync`)."
            ) from exc

        self._search_dates = search_dates
        self._settings = {
            "PREFER_DAY_OF_MONTH": "first",
            "PREFER_DATES_FROM": "past",
            "RETURN_AS_TIMEZONE_AWARE": False,
            "STRICT_PARSING": False,
        }

    def detect(self, text: str) -> list[Detection]:
        try:
            matches = self._search_dates(
                text,
                languages=self._languages,
                settings=self._settings,
            )
        except Exception:
            return []

        if not matches:
            return []

        findings: list[Detection] = []
        used: set[tuple[int, int]] = set()
        cursor = 0
        for item in matches:
            if not item or len(item) < 2:
                continue
            candidate = str(item[0]).strip()
            if not candidate:
                continue

            if len(candidate) < self._min_chars or len(candidate) > self._max_chars:
                continue
            if not any(char.isdigit() for char in candidate):
                continue
            if "@" in candidate:
                continue

            start = text.find(candidate, cursor)
            if start < 0:
                start = text.find(candidate)
            if start < 0:
                continue
            end = start + len(candidate)
            cursor = end

            key = (start, end)
            if key in used:
                continue
            used.add(key)

            findings.append(
                Detection(
                    start=start,
                    end=end,
                    text=text[start:end],
                    label=self._label,
                    score=self._score,
                    detector=self.name,
                    metadata={"source": "dateparser"},
                )
            )

        return sorted(findings, key=lambda item: (item.start, item.end))
