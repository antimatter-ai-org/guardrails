from __future__ import annotations


def _script_counts(text: str) -> dict[str, int]:
    cyrillic = 0
    latin = 0
    digits = 0
    for char in text:
        lowered = char.lower()
        if ("а" <= lowered <= "я") or lowered == "ё":
            cyrillic += 1
        elif "a" <= lowered <= "z":
            latin += 1
        elif char.isdigit():
            digits += 1
    return {
        "cyrillic": cyrillic,
        "latin": latin,
        "digits": digits,
    }


def classify_script_profile(text: str) -> str:
    counts = _script_counts(text)
    cyrillic = counts["cyrillic"]
    latin = counts["latin"]
    letters_total = cyrillic + latin
    if letters_total == 0:
        return "no_letters"

    cyr_share = cyrillic / letters_total
    lat_share = latin / letters_total
    if cyr_share >= 0.7:
        return "mostly_cyrillic"
    if lat_share >= 0.7:
        return "mostly_latin"
    return "mixed"


def resolve_languages(
    *,
    text: str,
    language_hint: str | None,
    supported: list[str],
    default_language: str,
    detection_mode: str,
    strategy: str,
    union_min_share: float,
) -> list[str]:
    supported_normalized = [item.strip().lower() for item in supported if item.strip()]
    if not supported_normalized:
        supported_normalized = ["en"]

    if language_hint:
        hinted = language_hint.strip().lower()
        if hinted in supported_normalized:
            return [hinted]

    if detection_mode == "hint_only":
        if default_language in supported_normalized:
            return [default_language]
        return [supported_normalized[0]]

    counts = _script_counts(text)
    cyrillic = counts["cyrillic"]
    latin = counts["latin"]
    letters_total = cyrillic + latin

    if letters_total == 0:
        if default_language in supported_normalized:
            return [default_language]
        return [supported_normalized[0]]

    cyr_share = cyrillic / letters_total
    lat_share = latin / letters_total
    preferred = "ru" if cyr_share >= lat_share else "en"
    if preferred not in supported_normalized:
        preferred = default_language if default_language in supported_normalized else supported_normalized[0]

    if strategy == "union":
        union_languages: list[str] = [preferred]
        if "ru" in supported_normalized and cyr_share >= float(union_min_share) and "ru" not in union_languages:
            union_languages.append("ru")
        if "en" in supported_normalized and lat_share >= float(union_min_share) and "en" not in union_languages:
            union_languages.append("en")
        if union_languages:
            return union_languages

    return [preferred]


def resolve_language(
    *,
    text: str,
    language_hint: str | None,
    supported: list[str],
    default_language: str,
    detection_mode: str,
) -> str:
    return resolve_languages(
        text=text,
        language_hint=language_hint,
        supported=supported,
        default_language=default_language,
        detection_mode=detection_mode,
        strategy="single",
        union_min_share=1.0,
    )[0]
