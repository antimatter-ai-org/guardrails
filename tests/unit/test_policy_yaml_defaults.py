from __future__ import annotations

from pathlib import Path

import yaml


def test_external_default_profile_includes_extended_regex_layers() -> None:
    policy = yaml.safe_load(Path("configs/policy.yaml").read_text(encoding="utf-8"))
    profile_name = policy["policies"]["external_default"]["analyzer_profile"]
    recognizers = policy["analyzer_profiles"][profile_name]["analysis"]["recognizers"]

    assert "identifier_regex" in recognizers
    assert "network_pii_regex" in recognizers
    assert "date_pii_regex" in recognizers


def test_network_and_date_recognizers_have_expected_labels() -> None:
    policy = yaml.safe_load(Path("configs/policy.yaml").read_text(encoding="utf-8"))
    definitions = policy["recognizer_definitions"]

    network_labels = {
        item["label"] for item in definitions["network_pii_regex"]["params"]["patterns"]
    }
    date_labels = {
        item["label"] for item in definitions["date_pii_regex"]["params"]["patterns"]
    }

    assert network_labels == {"IP_ADDRESS"}
    assert date_labels == {"DATE_TIME"}
