from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.config import RecognizerDefinition
from app.tools import download_models


def test_collect_token_classifier_models_from_policy(monkeypatch) -> None:
    config = SimpleNamespace(
        recognizer_definitions={
            "regex": RecognizerDefinition(type="regex", enabled=True, params={"patterns": []}),
            "nemotron": RecognizerDefinition(
                type="token_classifier",
                enabled=True,
                params={"model_name": "scanpatch/pii-ner-nemotron"},
            ),
            "other_token_model": RecognizerDefinition(
                type="token_classifier",
                enabled=True,
                params={"model_name": "acme/pii-token-classifier"},
            ),
        }
    )
    monkeypatch.setattr(download_models, "load_policy_config", lambda _: config)
    assert download_models._collect_token_classifier_models("unused") == [
        "acme/pii-token-classifier",
        "scanpatch/pii-ner-nemotron",
    ]


def test_download_models_run_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        download_models,
        "_collect_token_classifier_models",
        lambda _: ["scanpatch/pii-ner-nemotron"],
    )
    monkeypatch.setattr(download_models, "apply_model_env", lambda **_: None)

    def fake_download(*, output_dir: str, model_name: str, namespace: str) -> str:
        model_path = Path(output_dir) / namespace / model_name.replace("/", "__")
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "weights.bin").write_text("x", encoding="utf-8")
        return str(model_path)

    monkeypatch.setattr(download_models, "_download_hf_model", fake_download)

    exit_code = download_models.run(
        output_dir=str(tmp_path),
        policy_path="unused",
        extra_token_classifier_models=[],
    )

    assert exit_code == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["token_classifier_models"] == {
        "scanpatch/pii-ner-nemotron": str(tmp_path / "token_classifier" / "scanpatch__pii-ner-nemotron")
    }
    token_checksums = manifest["checksums"]["token_classifier_models"]["scanpatch/pii-ner-nemotron"]
    assert token_checksums["files"] == 1
    assert len(str(token_checksums["sha256_tree"])) == 64
