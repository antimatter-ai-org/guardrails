from __future__ import annotations

from app.config import DetectorDefinition, PolicyConfig, PolicyDefinition, RoutingConfig
from app.policy import PolicyResolver


def test_policy_resolver_uses_route_defaults() -> None:
    config = PolicyConfig(
        default_policy="external_default",
        routing=RoutingConfig(
            external_model_patterns=[r"^gpt-"],
            onprem_model_patterns=[r"^onprem/"],
            default_destination="external",
        ),
        policies={
            "external_default": PolicyDefinition(mode="mask", detectors=["a"]),
            "onprem_default": PolicyDefinition(mode="passthrough", detectors=[]),
        },
        detector_definitions={"a": DetectorDefinition(type="regex", enabled=True, params={"patterns": []})},
    )

    resolver = PolicyResolver(config)
    policy_name, policy = resolver.resolve_policy("onprem/mistral")
    assert policy_name == "onprem_default"
    assert policy.mode == "passthrough"

    policy_name2, policy2 = resolver.resolve_policy("gpt-4o-mini")
    assert policy_name2 == "external_default"
    assert policy2.mode == "mask"
