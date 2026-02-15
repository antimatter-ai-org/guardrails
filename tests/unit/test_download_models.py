from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.config import RecognizerDefinition
from app.tools import download_models


def test_collect_hf_token_classifier_models_skips_local_paths(tmp_path: Path, monkeypatch) -> None:
    local_model_dir = tmp_path / "local-model"
    local_model_dir.mkdir(parents=True)

    config = SimpleNamespace(
        recognizer_definitions={
            "remote_model": RecognizerDefinition(
                type="hf_token_classifier",
                enabled=True,
                params={"model_name": "dslim/bert-base-NER"},
            ),
            "local_model": RecognizerDefinition(
                type="hf_token_classifier",
                enabled=True,
                params={"model_name": str(local_model_dir)},
            ),
            "regex_model": RecognizerDefinition(type="regex", enabled=True, params={"patterns": []}),
        }
    )

    monkeypatch.setattr(download_models, "load_policy_config", lambda _: config)

    collected = download_models._collect_hf_token_classifier_models("unused")

    assert collected == ["dslim/bert-base-NER"]


def test_download_models_run_writes_token_classifier_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(download_models, "_collect_gliner_models", lambda _: ["urchade/gliner_multi-v2.1"])
    monkeypatch.setattr(download_models, "_collect_transformer_models", lambda _: [])
    monkeypatch.setattr(download_models, "_collect_hf_token_classifier_models", lambda _: ["dslim/bert-base-NER"])
    monkeypatch.setattr(download_models, "apply_model_env", lambda **_: None)

    def fake_download(*, output_dir: str, model_name: str, namespace: str) -> str:
        model_path = Path(output_dir) / namespace / model_name.replace("/", "__")
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "weights.bin").write_text("x", encoding="utf-8")
        return str(model_path)

    monkeypatch.setattr(download_models, "_download_hf_model", fake_download)

    exit_code = download_models.run(output_dir=str(tmp_path), policy_path="unused", extra_gliner_models=[])

    assert exit_code == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hf_token_classifier_models"] == {
        "dslim/bert-base-NER": str(tmp_path / "hf_token_classifier" / "dslim__bert-base-NER")
    }
    checksums = manifest["checksums"]["hf_token_classifier_models"]["dslim/bert-base-NER"]
    assert checksums["files"] == 1
    assert len(str(checksums["sha256_tree"])) == 64
