from __future__ import annotations

import re

from app.config import PolicyConfig, PolicyDefinition


class PolicyResolver:
    def __init__(self, config: PolicyConfig) -> None:
        self._config = config

    @property
    def config(self) -> PolicyConfig:
        return self._config

    def resolve_destination(self, model_name: str) -> str:
        routing = self._config.routing
        if any(re.search(pattern, model_name) for pattern in routing.onprem_model_patterns):
            return "onprem"
        if any(re.search(pattern, model_name) for pattern in routing.external_model_patterns):
            return "external"
        return routing.default_destination

    def resolve_policy(self, model_name: str, policy_name: str | None = None) -> tuple[str, PolicyDefinition]:
        if policy_name and policy_name in self._config.policies:
            return policy_name, self._config.policies[policy_name]

        destination = self.resolve_destination(model_name)
        explicit_name = f"{destination}_default"
        if explicit_name in self._config.policies:
            return explicit_name, self._config.policies[explicit_name]

        default_name = self._config.default_policy
        return default_name, self._config.policies[default_name]
