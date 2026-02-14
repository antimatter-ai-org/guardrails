from __future__ import annotations

from pathlib import Path

import yaml


def test_detector_documentation_covers_configured_detectors_and_labels() -> None:
    doc_path = Path("docs/DETECTORS.md")
    assert doc_path.exists(), "docs/DETECTORS.md must exist"
    doc_text = doc_path.read_text(encoding="utf-8")

    policy = yaml.safe_load(Path("configs/policy.yaml").read_text(encoding="utf-8"))
    detector_defs: dict[str, dict] = policy["detector_definitions"]

    for detector_name in detector_defs:
        assert f"## Detector: `{detector_name}`" in doc_text

    # Labels explicitly configured in YAML must be listed in docs.
    for detector_name, definition in detector_defs.items():
        detector_type = definition["type"]
        params = definition.get("params", {})

        if detector_type in {"regex", "secret_regex"}:
            patterns = params.get("patterns", [])
            for pattern in patterns:
                label = pattern["label"]
                assert f"`{label}`" in doc_text

    # Built-in / code-level labels that are not all in YAML must still be documented.
    built_in_labels = {
        "SECRET_AWS_ACCESS_KEY",
        "SECRET_GITHUB_TOKEN",
        "SECRET_SLACK_TOKEN",
        "SECRET_GENERIC_KEY",
        "SECRET_PRIVATE_KEY",
        "SECRET_HIGH_ENTROPY",
        "NER_PER",
        "NER_ORG",
        "NER_LOC",
    }
    for label in built_in_labels:
        assert f"`{label}`" in doc_text

    # GLiNER configured labels should appear with current emitted prefixing behavior.
    gliner = detector_defs.get("gliner_pii_multilingual")
    if gliner:
        for item in gliner.get("params", {}).get("labels", []):
            emitted = f"GLINER_{item}"
            assert f"`{emitted}`" in doc_text

def test_gliner_is_enabled_in_default_policy() -> None:
    policy = yaml.safe_load(Path("configs/policy.yaml").read_text(encoding="utf-8"))
    assert policy["detector_definitions"]["gliner_pii_multilingual"]["enabled"] is True
