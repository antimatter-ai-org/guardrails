from __future__ import annotations

from typing import Literal, get_args

# Canonical label taxonomy shared across runtime + evaluation.
# Keep this small and stable; extend intentionally as scope grows.

CanonicalPiiLabel = Literal[
    "person",
    "email",
    "phone",
    "location",
    "organization",
    "date",
    "ip",
    "identifier",
    "payment_card",
    "secret",
    "url",
]

CANONICAL_PII_LABELS: tuple[str, ...] = tuple(get_args(CanonicalPiiLabel))

