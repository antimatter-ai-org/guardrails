from __future__ import annotations

from pathlib import Path

import yaml


def test_detector_documentation_covers_configured_recognizers_and_labels() -> None:
    doc_path = Path("docs/DETECTORS.md")
    assert doc_path.exists(), "docs/DETECTORS.md must exist"
    doc_text = doc_path.read_text(encoding="utf-8")

    policy = yaml.safe_load(Path("configs/policy.yaml").read_text(encoding="utf-8"))
    recognizer_defs: dict[str, dict] = policy["recognizer_definitions"]

    for recognizer_name in recognizer_defs:
        assert f"## Recognizer: `{recognizer_name}`" in doc_text

    for definition in recognizer_defs.values():
        rec_type = definition["type"]
        params = definition.get("params", {})
        if rec_type in {"regex", "secret_regex"}:
            patterns = params.get("patterns", [])
            for pattern in patterns:
                label = pattern["label"]
                assert f"`{label}`" in doc_text


def test_gliner_is_enabled_in_default_profile() -> None:
    policy = yaml.safe_load(Path("configs/policy.yaml").read_text(encoding="utf-8"))
    profile_name = policy["policies"]["external_default"]["analyzer_profile"]
    recognizers = policy["analyzer_profiles"][profile_name]["analysis"]["recognizers"]
    assert "gliner_pii_multilingual" in recognizers
    assert policy["recognizer_definitions"]["gliner_pii_multilingual"]["enabled"] is True
