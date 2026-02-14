from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DetectorDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class PolicyDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Literal["mask", "passthrough", "block"] = "mask"
    detectors: list[str] = Field(default_factory=list)
    min_score: float = 0.5
    storage_ttl_seconds: int = 3600
    placeholder_prefix: str = "GR"


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: int = 1
    default_policy: str
    policies: dict[str, PolicyDefinition]
    detector_definitions: dict[str, DetectorDefinition]


def load_policy_config(path: str | Path) -> PolicyConfig:
    policy_path = Path(path)
    with policy_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return PolicyConfig.model_validate(raw)
