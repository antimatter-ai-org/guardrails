from __future__ import annotations

from app.core.analysis.language import classify_script_profile, resolve_language, resolve_languages


def test_classify_script_profile_detects_cyrillic() -> None:
    assert classify_script_profile("Привет как дела") == "mostly_cyrillic"


def test_classify_script_profile_detects_latin() -> None:
    assert classify_script_profile("hello this is english text") == "mostly_latin"


def test_classify_script_profile_detects_mixed() -> None:
    assert classify_script_profile("Привет hello") == "mixed"


def test_resolve_language_single_prefers_cyrillic() -> None:
    language = resolve_language(
        text="Привет world",
        language_hint=None,
        supported=["ru", "en"],
        default_language="ru",
        detection_mode="auto",
    )
    assert language == "ru"


def test_resolve_languages_union_returns_both_for_mixed_text() -> None:
    languages = resolve_languages(
        text="Привет hello world",
        language_hint=None,
        supported=["ru", "en"],
        default_language="ru",
        detection_mode="auto",
        strategy="union",
        union_min_share=0.1,
    )
    assert languages == ["en", "ru"]


def test_resolve_languages_respects_hint() -> None:
    languages = resolve_languages(
        text="Привет hello world",
        language_hint="ru",
        supported=["ru", "en"],
        default_language="en",
        detection_mode="auto",
        strategy="union",
        union_min_share=0.1,
    )
    assert languages == ["ru"]
