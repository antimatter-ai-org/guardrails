from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.config import RecognizerDefinition
from app.tools import download_models


def test_collect_gliner_models_from_policy(monkeypatch) -> None:
    config = SimpleNamespace(
        recognizer_definitions={
            "g1": RecognizerDefinition(
                type="gliner",
                enabled=True,
                params={"model_name": "urchade/gliner_multi-v2.1"},
            ),
            "g2": RecognizerDefinition(
                type="gliner",
                enabled=True,
                params={"model_name": "my-org/gliner-ru"},
            ),
            "regex": RecognizerDefinition(type="regex", enabled=True, params={"patterns": []}),
        }
    )
    monkeypatch.setattr(download_models, "load_policy_config", lambda _: config)
    assert download_models._collect_gliner_models("unused") == ["my-org/gliner-ru", "urchade/gliner_multi-v2.1"]


def test_download_models_run_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(download_models, "_collect_gliner_models", lambda _: ["urchade/gliner_multi-v2.1"])
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
    assert manifest["gliner_models"] == {
        "urchade/gliner_multi-v2.1": str(tmp_path / "gliner" / "urchade__gliner_multi-v2.1")
    }
    checksums = manifest["checksums"]["gliner_models"]["urchade/gliner_multi-v2.1"]
    assert checksums["files"] == 1
    assert len(str(checksums["sha256_tree"])) == 64
