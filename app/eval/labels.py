from __future__ import annotations


def canonicalize_prediction_label(label: str) -> str | None:
    normalized = label.strip().lower()

    if "email" in normalized:
        return "email"
    if "phone" in normalized:
        return "phone"
    if normalized in {"ner_per"} or "person" in normalized:
        return "person"
    if normalized in {"ner_org"} or "organization" in normalized or "org" in normalized:
        return "organization"
    if "ip" in normalized:
        return "ip"
    if "date" in normalized:
        return "date"
    if normalized in {"ner_loc"} or "location" in normalized or "address" in normalized:
        return "location"
    if "card" in normalized:
        return "payment_card"

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

    if "secret" in normalized or "api key" in normalized:
        return "secret"
    return None


def canonicalize_scanpatch_gold_label(label: str) -> str | None:
    normalized = label.strip().lower()

    if normalized in {
        "name",
        "first_name",
        "middle_name",
        "last_name",
        "name_initials",
        "nickname",
    }:
        return "person"
    if normalized == "organization":
        return "organization"
    if normalized.startswith("address"):
        return "location"
    if normalized == "mobile_phone":
        return "phone"
    if normalized == "email":
        return "email"
    if normalized == "ip":
        return "ip"
    if normalized == "date":
        return "date"
    if normalized in {
        "document_number",
        "military_individual_number",
        "vehicle_number",
        "tin",
        "snils",
    }:
        return "identifier"
    return None
