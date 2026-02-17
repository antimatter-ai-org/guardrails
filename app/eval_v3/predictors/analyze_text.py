from __future__ import annotations

from typing import Any

from app.eval.types import EvalSpan
from app.eval_v3.taxonomy import normalize_label


def canonicalize_prediction_label(label: str) -> str | None:
    raw = str(label or "").strip()
    if not raw:
        return None

    # Already canonical.
    canonical = normalize_label(raw)
    if canonical is not None:
        return canonical

    normalized = raw.strip().lower()

    # Presidio-style / common labels.
    if normalized in {"person", "name", "full name", "first name", "middle name", "last name"}:
        return "person"
    if normalized in {"email", "email_address", "e-mail"} or "email" in normalized:
        return "email"
    if normalized in {"phone", "phone_number", "phonenumber", "mobile_phone"} or "phone" in normalized:
        return "phone"
    if normalized in {"address", "location", "city", "district", "street", "postal_code", "zip", "zipcode"}:
        return "location"
    if "address" in normalized or "location" in normalized:
        return "location"
    if normalized in {"date", "date_time", "datetime"} or "date" in normalized:
        return "date"
    if normalized in {"credit_card", "credit_card_number", "card_number", "payment_card"} or "card" in normalized:
        return "payment_card"
    if normalized in {"ip_address", "ip"} or "ip" == normalized or normalized.startswith("ip_"):
        return "ip"
    if normalized in {"url", "uri", "domain"} or "url" in normalized:
        return "url"
    if normalized in {"organization", "org", "company"} or "organization" in normalized:
        return "organization"

    # Identifiers / documents.
    identifier_markers = (
        "passport",
        "document",
        "snils",
        "inn",
        "ogrn",
        "ssn",
        "iban",
        "swift",
        "tin",
        "vehicle",
        "military",
        "vin",
        "id",
    )
    if any(marker in normalized for marker in identifier_markers):
        return "identifier"

    # Secrets.
    if "secret" in normalized or "api_key" in normalized or "api key" in normalized or "token" in normalized or "jwt" in normalized:
        return "secret"

    return None


def as_eval_spans(detections: list[Any]) -> list[EvalSpan]:
    spans: list[EvalSpan] = []
    for item in detections:
        metadata = getattr(item, "metadata", None) or {}
        canonical = metadata.get("canonical_label")
        if canonical is None:
            canonical = canonicalize_prediction_label(getattr(item, "label", ""))
        else:
            canonical = normalize_label(str(canonical))

        spans.append(
            EvalSpan(
                start=int(getattr(item, "start")),
                end=int(getattr(item, "end")),
                label=str(getattr(item, "label", "")),
                canonical_label=canonical,
                score=getattr(item, "score", None),
                detector=getattr(item, "detector", None),
            )
        )
    return spans

