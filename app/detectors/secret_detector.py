from __future__ import annotations

from typing import Any

from app.detectors.regex_detector import RegexDetector

_DEFAULT_SECRET_PATTERNS: list[dict[str, Any]] = [
    {
        "name": "aws_access_key",
        "label": "SECRET_AWS_ACCESS_KEY",
        "pattern": r"\bAKIA[0-9A-Z]{16}\b",
        "score": 0.99,
    },
    {
        "name": "github_pat",
        "label": "SECRET_GITHUB_TOKEN",
        "pattern": r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b",
        "score": 0.99,
    },
    {
        "name": "slack_token",
        "label": "SECRET_SLACK_TOKEN",
        "pattern": r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
        "score": 0.99,
    },
    {
        "name": "generic_api_key",
        "label": "SECRET_GENERIC_KEY",
        "pattern": r"(?i)\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}['\"]?",
        "score": 0.9,
    },
    {
        "name": "private_key",
        "label": "SECRET_PRIVATE_KEY",
        "pattern": r"-----BEGIN (?:RSA|EC|OPENSSH|PGP|DSA) PRIVATE KEY-----",
        "score": 1.0,
    },
]


class SecretRegexDetector(RegexDetector):
    def __init__(self, name: str, patterns: list[dict[str, Any]] | None = None) -> None:
        merged = [*_DEFAULT_SECRET_PATTERNS]
        if patterns:
            merged.extend(patterns)
        super().__init__(name=name, patterns=merged)
