from __future__ import annotations

import json
from pathlib import Path

from app.runtime.mistral_regex_fix import (
    _MISTRAL_CORRECT_REGEX,
    _MISTRAL_INCORRECT_REGEX,
    maybe_fix_mistral_regex_in_tokenizer_dir,
)


def test_maybe_fix_mistral_regex_patches_tokenizer_json(tmp_path: Path) -> None:
    tokenizer_json = tmp_path / "tokenizer.json"
    payload = {
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": _MISTRAL_INCORRECT_REGEX},
                    "behavior": "Removed",
                }
            ],
        }
    }
    tokenizer_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert maybe_fix_mistral_regex_in_tokenizer_dir(tmp_path) is True

    updated = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    assert updated["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] == _MISTRAL_CORRECT_REGEX


def test_maybe_fix_mistral_regex_noop_when_missing(tmp_path: Path) -> None:
    assert maybe_fix_mistral_regex_in_tokenizer_dir(tmp_path) is False

