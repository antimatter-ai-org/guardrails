from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.model_assets import (
    apply_model_env,
    gliner_local_dir,
    gliner_repo_dirname,
    resolve_gliner_model_source,
    resolve_token_classifier_model_source,
    token_classifier_local_dir,
    token_classifier_repo_dirname,
)


def test_gliner_repo_dirname() -> None:
    assert gliner_repo_dirname("urchade/gliner_multi-v2.1") == "urchade__gliner_multi-v2.1"


def test_gliner_local_dir() -> None:
    assert gliner_local_dir("/tmp/models", "urchade/gliner_multi-v2.1") == Path(
        "/tmp/models/gliner/urchade__gliner_multi-v2.1"
    )


def test_token_classifier_repo_dirname() -> None:
    assert token_classifier_repo_dirname("scanpatch/pii-ner-nemotron") == "scanpatch__pii-ner-nemotron"


def test_token_classifier_local_dir() -> None:
    assert token_classifier_local_dir("/tmp/models", "scanpatch/pii-ner-nemotron") == Path(
        "/tmp/models/token_classifier/scanpatch__pii-ner-nemotron"
    )


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


def test_resolve_gliner_model_source_strict_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_gliner_model_source("urchade/gliner_multi-v2.1", str(tmp_path), strict=True)


def test_resolve_token_classifier_model_source_with_model_dir(tmp_path: Path) -> None:
    model_name = "scanpatch/pii-ner-nemotron"
    local_dir = token_classifier_local_dir(str(tmp_path), model_name)
    local_dir.mkdir(parents=True)
    assert resolve_token_classifier_model_source(model_name, str(tmp_path)) == str(local_dir)


def test_resolve_token_classifier_model_source_strict_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_token_classifier_model_source("scanpatch/pii-ner-nemotron", str(tmp_path), strict=True)


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
