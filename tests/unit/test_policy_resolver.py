from __future__ import annotations

import pytest

from app.config import AnalysisConfig, AnalyzerProfile, PolicyConfig, PolicyDefinition, RecognizerDefinition
from app.policy import PolicyResolver


def _config() -> PolicyConfig:
    return PolicyConfig(
        default_policy="external",
        policies={
            "external": PolicyDefinition(mode="mask", analyzer_profile="default_profile"),
            "external_blocking": PolicyDefinition(mode="block", analyzer_profile="default_profile"),
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
    assert default_name == "external"
    assert default_policy.mode == "mask"

    strict_name, strict_policy = resolver.resolve_policy("external_blocking")
    assert strict_name == "external_blocking"
    assert strict_policy.mode == "block"


def test_policy_resolver_raises_on_unknown_policy() -> None:
    resolver = PolicyResolver(_config())

    with pytest.raises(KeyError):
        resolver.resolve_policy("missing")
