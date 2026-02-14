from __future__ import annotations

import pytest

from app.config import AnalysisConfig, AnalyzerProfile, PolicyConfig, PolicyDefinition, RecognizerDefinition
from app.policy import PolicyResolver


def _config() -> PolicyConfig:
    return PolicyConfig(
        default_policy="external_default",
        policies={
            "external_default": PolicyDefinition(mode="mask", analyzer_profile="default_profile"),
            "strict_block": PolicyDefinition(mode="block", analyzer_profile="default_profile"),
        },
        analyzer_profiles={
            "default_profile": AnalyzerProfile(
                analysis=AnalysisConfig(recognizers=["a"]),
            )
        },
        recognizer_definitions={
            "a": RecognizerDefinition(type="regex", enabled=True, params={"patterns": []})
        },
    )


def test_policy_resolver_uses_default_or_explicit_policy() -> None:
    resolver = PolicyResolver(_config())

    default_name, default_policy = resolver.resolve_policy()
    assert default_name == "external_default"
    assert default_policy.mode == "mask"

    strict_name, strict_policy = resolver.resolve_policy("strict_block")
    assert strict_name == "strict_block"
    assert strict_policy.mode == "block"


def test_policy_resolver_raises_on_unknown_policy() -> None:
    resolver = PolicyResolver(_config())

    with pytest.raises(KeyError):
        resolver.resolve_policy("missing")
