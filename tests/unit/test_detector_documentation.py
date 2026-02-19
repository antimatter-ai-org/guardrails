from __future__ import annotations

from pathlib import Path

import yaml


def test_architecture_docs_list_all_configured_recognizers() -> None:
    doc_path = Path("docs/ARCHITECTURE.md")
    assert doc_path.exists(), "docs/ARCHITECTURE.md must exist"
    doc_text = doc_path.read_text(encoding="utf-8")

    policy = yaml.safe_load(Path("configs/policy.yaml").read_text(encoding="utf-8"))
    recognizer_defs: dict[str, dict] = policy["recognizer_definitions"]

    for recognizer_name in recognizer_defs:
        assert f"`{recognizer_name}`" in doc_text


def test_architecture_docs_cover_regex_labels() -> None:
    doc_text = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    policy = yaml.safe_load(Path("configs/policy.yaml").read_text(encoding="utf-8"))

    for definition in policy["recognizer_definitions"].values():
        rec_type = definition["type"]
        params = definition.get("params", {})
        if rec_type in {"regex", "secret_regex"}:
            for pattern in params.get("patterns", []):
                assert f"`{pattern['label']}`" in doc_text
