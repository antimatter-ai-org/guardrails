from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RecognizerDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["regex", "secret_regex", "phone", "ip", "gliner", "token_classifier", "entropy", "natasha_ner"]
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    recognizers: list[str] = Field(default_factory=list)


class AnalyzerProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


class PolicyDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Literal["mask", "passthrough", "block"] = "mask"
    analyzer_profile: str
    min_score: float = 0.5
    storage_ttl_seconds: int = 3600
    placeholder_prefix: str = "GR"


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    default_policy: str
    policies: dict[str, PolicyDefinition]
    analyzer_profiles: dict[str, AnalyzerProfile]
    recognizer_definitions: dict[str, RecognizerDefinition]

    @model_validator(mode="after")
    def _validate_references(self) -> "PolicyConfig":
        if self.default_policy not in self.policies:
            raise ValueError(f"default_policy '{self.default_policy}' is not defined in policies")

        for policy_name, policy in self.policies.items():
            if policy.analyzer_profile not in self.analyzer_profiles:
                raise ValueError(
                    f"policy '{policy_name}' references missing analyzer_profile '{policy.analyzer_profile}'"
                )

        for profile_name, profile in self.analyzer_profiles.items():
            missing = [name for name in profile.analysis.recognizers if name not in self.recognizer_definitions]
            if missing:
                raise ValueError(
                    f"analyzer_profile '{profile_name}' references missing recognizers: {', '.join(missing)}"
                )

        return self


def load_policy_config(path: str | Path) -> PolicyConfig:
    policy_path = Path(path)
    with policy_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return PolicyConfig.model_validate(raw)
