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
            # Respect explicit environment configuration (e.g. separate dataset/model caches
            # on remote hosts). Model code can still load from GR_MODEL_DIR via local paths.
            os.environ.setdefault("HF_HOME", str(hf_cache))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))

    if offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def token_classifier_repo_dirname(model_name: str) -> str:
    return model_name.replace("/", "__")


def token_classifier_local_dir(model_dir: str, model_name: str) -> Path:
    return Path(model_dir) / "token_classifier" / token_classifier_repo_dirname(model_name)


def _resolve_model_source(
    *,
    model_name: str,
    model_dir: str | None,
    namespace_dir: str,
    strict: bool,
    error_prefix: str,
) -> str:
    if not model_dir:
        return model_name

    local_dir = Path(model_dir) / namespace_dir / model_name.replace("/", "__")
    if local_dir.exists():
        return str(local_dir)

    explicit = Path(model_name)
    if explicit.exists():
        return str(explicit)

    if strict:
        raise FileNotFoundError(f"{error_prefix} model not found in model_dir: {local_dir}")
    return model_name


def resolve_token_classifier_model_source(model_name: str, model_dir: str | None, *, strict: bool = False) -> str:
    return _resolve_model_source(
        model_name=model_name,
        model_dir=model_dir,
        namespace_dir="token_classifier",
        strict=strict,
        error_prefix="Token classifier",
    )
