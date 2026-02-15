from __future__ import annotations


def _script_counts(text: str) -> dict[str, int]:
    cyrillic = 0
    latin = 0
    for char in text:
        lowered = char.lower()
        if ("а" <= lowered <= "я") or lowered == "ё":
            cyrillic += 1
        elif "a" <= lowered <= "z":
            latin += 1
    return {
        "cyrillic": cyrillic,
        "latin": latin,
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
