from __future__ import annotations

import json

from app.runtime.mistral_regex_fix import (
    _MISTRAL_INCORRECT_REGEX,
    _MISTRAL_MODEL_TYPES,
    maybe_fix_mistral_regex_in_tokenizer_dir,
)


def test_patches_false_mistral_warning_for_non_mistral_config(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "xlm-roberta", "transformers_version": "4.57.3"}, ensure_ascii=False),
        encoding="utf-8",
    )
    # tokenizer.json is optional for this path: config patch should still apply.
    changed = maybe_fix_mistral_regex_in_tokenizer_dir(tmp_path)
    assert changed is True
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["transformers_version"] == "4.57.4"


def test_does_not_patch_mistral_model_types(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": next(iter(_MISTRAL_MODEL_TYPES)), "transformers_version": "4.57.3"}, ensure_ascii=False),
        encoding="utf-8",
    )
    changed = maybe_fix_mistral_regex_in_tokenizer_dir(tmp_path)
    assert changed is False
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["transformers_version"] == "4.57.3"


def test_patches_incorrect_regex_pattern_in_tokenizer_json(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "mistral", "transformers_version": "4.57.3"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "pre_tokenizer": {
                    "type": "Split",
                    "pattern": {"Regex": _MISTRAL_INCORRECT_REGEX},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    changed = maybe_fix_mistral_regex_in_tokenizer_dir(tmp_path)
    assert changed is True
    tok = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    assert tok["pre_tokenizer"]["pattern"]["Regex"] != _MISTRAL_INCORRECT_REGEX

