from __future__ import annotations

from dataclasses import dataclass


CANONICAL_LABELS: tuple[str, ...] = (
    "person",
    "email",
    "phone",
    "location",
    "date",
    "identifier",
    "payment_card",
    "ip",
    "url",
    "secret",
    "organization",
)


# Default risk weights used for headline metrics (configurable in code later if needed).
# Higher weight means higher impact if missed (FN / leakage).
DEFAULT_RISK_WEIGHTS: dict[str, float] = {
    "secret": 5.0,
    "payment_card": 5.0,
    "identifier": 4.0,
    "email": 3.0,
    "phone": 3.0,
    "ip": 3.0,
    "url": 2.0,
    "person": 2.0,
    "organization": 2.0,
    "location": 1.0,
    "date": 1.0,
}


def normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    if not label:
        return None
    if label in CANONICAL_LABELS:
        return label
    return None


@dataclass(frozen=True, slots=True)
class WeightedValue:
    value: float
    weight: float


def weighted_average(items: list[WeightedValue]) -> float:
    denom = sum(item.weight for item in items)
    if denom <= 0:
        return 0.0
    return sum(item.value * item.weight for item in items) / denom

