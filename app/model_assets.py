from __future__ import annotations

import os
from pathlib import Path


def _normalize_model_dir(model_dir: str | None) -> Path | None:
    if not model_dir:
        return None
    return Path(model_dir).expanduser().resolve()


def apply_model_env(model_dir: str | None, offline_mode: bool) -> None:
    base_dir = _normalize_model_dir(model_dir)
    if base_dir is not None:
        hf_cache = base_dir / "hf_cache"
        if not hf_cache.exists():
            try:
                hf_cache.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        if hf_cache.exists():
            os.environ["HF_HOME"] = str(hf_cache)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache / "hub")
            os.environ["TRANSFORMERS_CACHE"] = str(hf_cache / "transformers")

    if offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def gliner_repo_dirname(model_name: str) -> str:
    return model_name.replace("/", "__")


def gliner_local_dir(model_dir: str, model_name: str) -> Path:
    return Path(model_dir) / "gliner" / gliner_repo_dirname(model_name)


def resolve_gliner_model_source(model_name: str, model_dir: str | None, *, strict: bool = False) -> str:
    if not model_dir:
        return model_name

    local_dir = gliner_local_dir(model_dir, model_name)
    if local_dir.exists():
        return str(local_dir)

    explicit = Path(model_name)
    if explicit.exists():
        return str(explicit)

    if strict:
        raise FileNotFoundError(f"GLiNER model not found in model_dir: {local_dir}")
    return model_name
