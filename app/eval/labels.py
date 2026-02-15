from __future__ import annotations


def canonicalize_prediction_label(label: str) -> str | None:
    normalized = label.strip().lower()

    if normalized in {"full name", "first name", "middle name", "last name", "name", "name initials", "nickname"}:
        return "person"
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
    if normalized in {
        "ner_loc",
        "city",
        "district",
        "street address",
        "postal code",
        "street",
        "region",
        "country",
        "house",
        "building",
        "apartment",
        "geolocation",
    }:
        return "location"
    if "location" in normalized or "address" in normalized:
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


def canonicalize_rubai_gold_label(label: str) -> str | None:
    normalized = label.strip().lower()
    if normalized in {"text", "o", ""}:
        return None
    if normalized in {"name", "first_name", "last_name", "full_name"}:
        return "person"
    if normalized in {"phone", "phone_number", "mobile_phone"}:
        return "phone"
    if normalized in {"address", "location", "city", "district", "street", "postal_code"}:
        return "location"
    if normalized in {"date", "datetime"}:
        return "date"
    if normalized in {"card_number", "credit_card", "payment_card", "card"}:
        return "payment_card"
    if normalized in {"document_id", "document_number", "passport", "tin", "snils", "id"}:
        return "identifier"
    if normalized in {"email"}:
        return "email"
    return None
