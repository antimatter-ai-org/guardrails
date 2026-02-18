from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# See: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84
#
# Some tokenizer.json bundles ship with an incorrect pre-tokenizer regex, which triggers
# runtime warnings and can lead to incorrect tokenization. Newer transformers exposes a
# `fix_mistral_regex=True` flag, but our pinned transformers may not support it yet.
#
# We patch local tokenizer.json files (when loading from a directory) to ensure deterministic,
# warning-free tokenization across environments.

_MISTRAL_INCORRECT_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|\s+(?!\S)|\s+"
)

_MISTRAL_CORRECT_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*|"
    r"\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _walk_and_patch_regex(obj: Any) -> bool:
    changed = False
    if isinstance(obj, dict):
        # tokenizers JSON stores regex patterns as: {"pattern": {"Regex": "..."}}
        pat = obj.get("pattern")
        if isinstance(pat, dict) and isinstance(pat.get("Regex"), str):
            if pat["Regex"] == _MISTRAL_INCORRECT_REGEX:
                pat["Regex"] = _MISTRAL_CORRECT_REGEX
                changed = True
        for value in obj.values():
            changed = _walk_and_patch_regex(value) or changed
    elif isinstance(obj, list):
        for item in obj:
            changed = _walk_and_patch_regex(item) or changed
    return changed


def maybe_fix_mistral_regex_in_tokenizer_dir(model_dir: str | Path) -> bool:
    """Best-effort patch for local model directories containing tokenizer.json.

    Returns True if a patch was applied.
    """

    path = Path(model_dir)
    if not path.exists() or not path.is_dir():
        return False
    tokenizer_json = path / "tokenizer.json"
    if not tokenizer_json.exists():
        return False

    try:
        data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    except Exception:
        return False

    changed = _walk_and_patch_regex(data)
    if not changed:
        return False

    tmp = tokenizer_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(tokenizer_json)
    return True
