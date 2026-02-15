from __future__ import annotations

from app.eval.script_profile import classify_script_profile


def test_classify_script_profile_detects_cyrillic() -> None:
    assert classify_script_profile("Привет как дела") == "mostly_cyrillic"


def test_classify_script_profile_detects_latin() -> None:
    assert classify_script_profile("hello this is english text") == "mostly_latin"


def test_classify_script_profile_detects_mixed() -> None:
    assert classify_script_profile("Привет hello") == "mixed"


def test_classify_script_profile_handles_no_letters() -> None:
    assert classify_script_profile("12345 !!!") == "no_letters"
