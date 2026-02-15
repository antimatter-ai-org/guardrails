from __future__ import annotations

import os
import shutil
from pathlib import Path

NATASHA_EMBED_FILENAME = "navec_news_v1_1B_250K_300d_100q.tar"
NATASHA_NER_FILENAME = "slovnet_ner_news_v1.tar"


def _normalize_model_dir(model_dir: str | None) -> Path | None:
    if not model_dir:
        return None
    path = Path(model_dir).expanduser().resolve()
    return path


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


def hf_model_local_dir(model_dir: str, namespace: str, model_name: str) -> Path:
    return Path(model_dir) / namespace / model_name.replace("/", "__")


def resolve_hf_model_source(
    *,
    model_name: str,
    model_dir: str | None,
    namespace: str,
    strict: bool = False,
) -> str:
    if not model_dir:
        return model_name

    local_dir = hf_model_local_dir(model_dir, namespace, model_name)
    if local_dir.exists():
        return str(local_dir)

    explicit = Path(model_name)
    if explicit.exists():
        return str(explicit)

    if strict:
        raise FileNotFoundError(f"Model not found in model_dir: {local_dir}")
    return model_name


def natasha_local_paths(model_dir: str | None, *, strict: bool = False) -> tuple[str | None, str | None]:
    base_dir = _normalize_model_dir(model_dir)
    if base_dir is None:
        return None, None

    emb_path = base_dir / "natasha" / "emb" / NATASHA_EMBED_FILENAME
    ner_path = base_dir / "natasha" / "model" / NATASHA_NER_FILENAME
    if emb_path.exists() and ner_path.exists():
        return str(emb_path), str(ner_path)

    if strict:
        raise FileNotFoundError(
            f"Natasha model files not found in model_dir: {emb_path} and {ner_path}"
        )
    return None, None


def copy_natasha_models(output_dir: str) -> dict[str, str]:
    from natasha.emb import NEWS_EMBEDDING
    from natasha.ner import NEWS_NER

    root = Path(output_dir)
    emb_dest = root / "natasha" / "emb" / NATASHA_EMBED_FILENAME
    ner_dest = root / "natasha" / "model" / NATASHA_NER_FILENAME
    emb_dest.parent.mkdir(parents=True, exist_ok=True)
    ner_dest.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(NEWS_EMBEDDING, emb_dest)
    shutil.copy2(NEWS_NER, ner_dest)

    return {"embedding": str(emb_dest), "ner": str(ner_dest)}
