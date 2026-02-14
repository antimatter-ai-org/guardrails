from __future__ import annotations

import pytest

from app.config import DetectorDefinition, PolicyConfig, PolicyDefinition
from app.policy import PolicyResolver


def test_policy_resolver_uses_default_or_explicit_policy() -> None:
    config = PolicyConfig(
        default_policy="external_default",
        policies={
            "external_default": PolicyDefinition(mode="mask", detectors=["a"]),
            "strict_block": PolicyDefinition(mode="block", detectors=["a"]),
        },
        detector_definitions={"a": DetectorDefinition(type="regex", enabled=True, params={"patterns": []})},
    )

    resolver = PolicyResolver(config)
    default_name, default_policy = resolver.resolve_policy()
    assert default_name == "external_default"
    assert default_policy.mode == "mask"

    strict_name, strict_policy = resolver.resolve_policy("strict_block")
    assert strict_name == "strict_block"
    assert strict_policy.mode == "block"


def test_policy_resolver_raises_on_unknown_policy() -> None:
    config = PolicyConfig(
        default_policy="external_default",
        policies={"external_default": PolicyDefinition(mode="mask", detectors=[])},
        detector_definitions={},
    )
    resolver = PolicyResolver(config)

    with pytest.raises(KeyError):
        resolver.resolve_policy("missing")
