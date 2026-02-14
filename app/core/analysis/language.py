from __future__ import annotations


def resolve_language(
    *,
    text: str,
    language_hint: str | None,
    supported: list[str],
    default_language: str,
    detection_mode: str,
) -> str:
    supported_normalized = [item.strip().lower() for item in supported if item.strip()]
    if not supported_normalized:
        supported_normalized = ["en"]

    if language_hint:
        hinted = language_hint.strip().lower()
        if hinted in supported_normalized:
            return hinted

    if detection_mode == "hint_only":
        if default_language in supported_normalized:
            return default_language
        return supported_normalized[0]

    has_cyrillic = any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in text)
    preferred = "ru" if has_cyrillic else "en"
    if preferred in supported_normalized:
        return preferred

    if default_language in supported_normalized:
        return default_language
    return supported_normalized[0]
