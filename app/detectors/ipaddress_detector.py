from __future__ import annotations

import ipaddress

import regex as re

from app.detectors.base import Detector
from app.models.entities import Detection


class IPAddressDetector(Detector):
    def __init__(
        self,
        name: str,
        score: float = 0.995,
        label: str = "IP_ADDRESS_LIB",
    ) -> None:
        super().__init__(name)
        self._score = float(score)
        self._label = label
        self._candidate_re = re.compile(
            r"(?<![A-Za-z0-9_])(?:[0-9A-Fa-f:.]{2,}(?:/[0-9]{1,3})?)(?![A-Za-z0-9_])"
        )

    @staticmethod
    def _trim_token(raw: str) -> tuple[str, int, int]:
        trim_chars = "[](){}<>,;\"'`"
        left = 0
        right = len(raw)
        while left < right and raw[left] in trim_chars:
            left += 1
        while right > left and raw[right - 1] in trim_chars:
            right -= 1
        return raw[left:right], left, len(raw) - right

    @staticmethod
    def _is_ip_candidate(token: str) -> bool:
        separators = token.count(".") + token.count(":")
        if separators < 2:
            return False
        try:
            if "/" in token:
                ipaddress.ip_network(token, strict=False)
            else:
                ipaddress.ip_address(token)
        except Exception:
            return False
        return True

    def detect(self, text: str) -> list[Detection]:
        findings: list[Detection] = []
        seen: set[tuple[int, int]] = set()

        for match in self._candidate_re.finditer(text):
            raw = match.group(0)
            token, left_trim, right_trim = self._trim_token(raw)
            if not token:
                continue
            if not self._is_ip_candidate(token):
                continue

            start = match.start() + left_trim
            end = match.end() - right_trim
            if end <= start:
                continue
            if (start, end) in seen:
                continue
            seen.add((start, end))
            findings.append(
                Detection(
                    start=start,
                    end=end,
                    text=text[start:end],
                    label=self._label,
                    score=self._score,
                    detector=self.name,
                    metadata={"source": "ipaddress"},
                )
            )

        return findings
