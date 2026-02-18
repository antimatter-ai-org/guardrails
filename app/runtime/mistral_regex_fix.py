from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# See: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84
#
# Some tokenizer.json bundles ship with an incorrect pre-tokenizer regex, which triggers
# runtime warnings and can lead to incorrect tokenization.
#
# In transformers v5.1.0, we also observed a false-positive "Mistral regex" warning for
# non-mistral local tokenizers when `config.json` contains `transformers_version=4.57.3`.
# We patch the local cached config to avoid that warning without changing tokenization.
#
# We patch local tokenizer.json files (when loading from a directory) to ensure deterministic,
# warning-free tokenization across environments.

_MISTRAL_MODEL_TYPES = {"mistral", "mistral3", "voxtral", "ministral", "pixtral"}

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


def _maybe_fix_transformers_false_mistral_warning(model_dir: Path) -> bool:
    """
    Work around a false-positive warning in transformers v5.1.0:

    TokenizersBackend._patch_mistral_regex() warns for *any* local tokenizer config with
    transformers_version == 4.57.3 (even when model_type is not mistral), because the
    early-return logic only triggers for <=4.57.2 or >4.57.3.

    For non-mistral model types, we bump transformers_version to 4.57.4 in the cached
    local config.json so transformers treats it as "fixed" and skips the mistral path.
    """

    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    model_type = str(data.get("model_type") or "").strip().lower()
    transformers_version = str(data.get("transformers_version") or "").strip()
    if not model_type or not transformers_version:
        return False
    if model_type in _MISTRAL_MODEL_TYPES:
        return False
    if transformers_version != "4.57.3":
        return False

    data["transformers_version"] = "4.57.4"
    tmp = config_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(config_path)
    return True


def maybe_fix_mistral_regex_in_tokenizer_dir(model_dir: str | Path) -> bool:
    """Best-effort patch for local model directories containing tokenizer.json.

    Returns True if a patch was applied.
    """

    path = Path(model_dir)
    if not path.exists() or not path.is_dir():
        return False
    changed_any = False

    # Avoid spurious mistral warnings for non-mistral configs in some transformers versions.
    changed_any = _maybe_fix_transformers_false_mistral_warning(path) or changed_any

    tokenizer_json = path / "tokenizer.json"
    if not tokenizer_json.exists():
        return changed_any

    try:
        data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    except Exception:
        return changed_any

    changed = _walk_and_patch_regex(data)
    if not changed:
        return changed_any

    tmp = tokenizer_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(tokenizer_json)
    return True
