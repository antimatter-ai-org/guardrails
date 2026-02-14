from __future__ import annotations

from app.config import PolicyConfig, PolicyDefinition


class PolicyResolver:
    def __init__(self, config: PolicyConfig) -> None:
        self._config = config

    @property
    def config(self) -> PolicyConfig:
        return self._config

    def resolve_policy(self, policy_name: str | None = None) -> tuple[str, PolicyDefinition]:
        if policy_name is not None:
            policy = self._config.policies.get(policy_name)
            if policy is None:
                raise KeyError(f"unknown policy: {policy_name}")
            return policy_name, policy

        default_name = self._config.default_policy
        policy = self._config.policies.get(default_name)
        if policy is None:
            raise KeyError(f"default policy '{default_name}' is not configured")
        return default_name, policy

    def list_policies(self) -> list[str]:
        return sorted(self._config.policies.keys())
