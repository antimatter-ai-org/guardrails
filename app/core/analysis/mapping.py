from __future__ import annotations


def canonicalize_entity_type(entity_type: str, custom_mapping: dict[str, str] | None = None) -> str | None:
    normalized = entity_type.strip().upper()
    if not normalized:
        return None

    if custom_mapping:
        candidate = custom_mapping.get(normalized)
        if candidate:
            return candidate

    direct = {
        "PERSON": "person",
        "PER": "person",
        "LOCATION": "location",
        "LOC": "location",
        "ORGANIZATION": "organization",
        "ORG": "organization",
        "EMAIL_ADDRESS": "email",
        "EMAIL": "email",
        "PHONE_NUMBER": "phone",
        "PHONE": "phone",
        "IP_ADDRESS": "ip",
        "IP": "ip",
        "DATE_TIME": "date",
        "DATE": "date",
        "URL": "url",
        "CREDIT_CARD": "payment_card",
        "PAYMENT_CARD": "payment_card",
        "SECRET": "secret",
        "API_KEY": "secret",
        "US_SSN": "identifier",
        "IBAN_CODE": "identifier",
        "SWIFT_CODE": "identifier",
        "PASSPORT": "identifier",
        "DOCUMENT_NUMBER": "identifier",
        "SNILS": "identifier",
        "TIN": "identifier",
        "VEHICLE_NUMBER": "identifier",
        "MILITARY_INDIVIDUAL_NUMBER": "identifier",
    }
    if normalized in direct:
        return direct[normalized]

    if "EMAIL" in normalized:
        return "email"
    if "PHONE" in normalized:
        return "phone"
    if "IP" in normalized:
        return "ip"
    if "DATE" in normalized:
        return "date"
    if "URL" in normalized:
        return "url"
    if "PERSON" in normalized or normalized == "PER":
        return "person"
    if "ORG" in normalized:
        return "organization"
    if "LOC" in normalized or "ADDRESS" in normalized or "CITY" in normalized:
        return "location"
    if "CARD" in normalized:
        return "payment_card"
    if "SECRET" in normalized or "KEY" in normalized:
        return "secret"
    return None
