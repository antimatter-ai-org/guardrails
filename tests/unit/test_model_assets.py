from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.model_assets import (
    NATASHA_EMBED_FILENAME,
    NATASHA_NER_FILENAME,
    apply_model_env,
    gliner_repo_dirname,
    gliner_local_dir,
    hf_model_local_dir,
    natasha_local_paths,
    resolve_hf_model_source,
    resolve_gliner_model_source,
)


def test_gliner_repo_dirname() -> None:
    assert gliner_repo_dirname("urchade/gliner_multi-v2.1") == "urchade__gliner_multi-v2.1"


def test_resolve_gliner_model_source_without_model_dir() -> None:
    assert resolve_gliner_model_source("urchade/gliner_multi-v2.1", None) == "urchade/gliner_multi-v2.1"


def test_resolve_gliner_model_source_with_model_dir(tmp_path: Path) -> None:
    model_name = "urchade/gliner_multi-v2.1"
    local_dir = gliner_local_dir(str(tmp_path), model_name)
    local_dir.mkdir(parents=True)

    assert resolve_gliner_model_source(model_name, str(tmp_path)) == str(local_dir)


def test_resolve_gliner_model_source_non_strict_fallback(tmp_path: Path) -> None:
    model_name = "urchade/gliner_multi-v2.1"
    assert resolve_gliner_model_source(model_name, str(tmp_path), strict=False) == model_name


def test_hf_model_local_dir() -> None:
    assert hf_model_local_dir("/tmp/models", "hf_token_classifier", "dslim/bert-base-NER") == Path(
        "/tmp/models/hf_token_classifier/dslim__bert-base-NER"
    )


def test_resolve_hf_model_source_with_model_dir(tmp_path: Path) -> None:
    model_name = "dslim/bert-base-NER"
    local_dir = hf_model_local_dir(str(tmp_path), "hf_token_classifier", model_name)
    local_dir.mkdir(parents=True)

    assert (
        resolve_hf_model_source(model_name=model_name, model_dir=str(tmp_path), namespace="hf_token_classifier")
        == str(local_dir)
    )


def test_resolve_hf_model_source_non_strict_fallback(tmp_path: Path) -> None:
    model_name = "dslim/bert-base-NER"
    assert (
        resolve_hf_model_source(
            model_name=model_name,
            model_dir=str(tmp_path),
            namespace="hf_token_classifier",
            strict=False,
        )
        == model_name
    )


def test_resolve_hf_model_source_strict_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_hf_model_source(
            model_name="dslim/bert-base-NER",
            model_dir=str(tmp_path),
            namespace="hf_token_classifier",
            strict=True,
        )


def test_natasha_local_paths(tmp_path: Path) -> None:
    emb_path, ner_path = natasha_local_paths(str(tmp_path))
    assert emb_path is None
    assert ner_path is None

    local_emb = tmp_path / "natasha" / "emb" / NATASHA_EMBED_FILENAME
    local_ner = tmp_path / "natasha" / "model" / NATASHA_NER_FILENAME
    local_emb.parent.mkdir(parents=True)
    local_ner.parent.mkdir(parents=True)
    local_emb.write_text("x", encoding="utf-8")
    local_ner.write_text("x", encoding="utf-8")

    emb_path, ner_path = natasha_local_paths(str(tmp_path))
    assert emb_path is not None and emb_path.endswith(f"/natasha/emb/{NATASHA_EMBED_FILENAME}")
    assert ner_path is not None and ner_path.endswith(f"/natasha/model/{NATASHA_NER_FILENAME}")


def test_natasha_local_paths_strict_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        natasha_local_paths(str(tmp_path), strict=True)


def test_apply_model_env_sets_offline_and_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    apply_model_env(model_dir=str(tmp_path), offline_mode=True)

    assert (tmp_path / "hf_cache").exists()
    assert "hf_cache" in str(Path(os.getenv("HF_HOME", "")))
    assert os.getenv("HF_HUB_OFFLINE") == "1"
    assert os.getenv("TRANSFORMERS_OFFLINE") == "1"
