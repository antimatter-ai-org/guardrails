from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TextItem(BaseModel):
    id: str
    text: str


class DetectionItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    start: int
    end: int
    label: str
    score: float
    detector: str
    snippet: str
    entity_type: str | None = None
    canonical_label: str | None = None


class DetectRequest(BaseModel):
    policy_name: str | None = None
    items: list[TextItem] = Field(min_length=1)


class DetectResultItem(BaseModel):
    id: str
    detections: list[DetectionItem]
    diagnostics: dict[str, object] = Field(default_factory=dict)


class DetectResponse(BaseModel):
    policy_name: str
    mode: Literal["mask", "passthrough", "block"]
    analysis_backend: Literal["presidio"] = "presidio"
    findings_count: int
    items: list[DetectResultItem]


class MaskRequest(BaseModel):
    request_id: str
    policy_name: str | None = None
    items: list[TextItem] = Field(min_length=1)
    store_context: bool = True


class MaskResultItem(BaseModel):
    id: str
    text: str
    detections: list[DetectionItem]
    diagnostics: dict[str, object] = Field(default_factory=dict)


class MaskResponse(BaseModel):
    request_id: str
    policy_name: str
    mode: Literal["mask", "passthrough", "block"]
    analysis_backend: Literal["presidio"] = "presidio"
    findings_count: int
    placeholders_count: int
    context_stored: bool
    items: list[MaskResultItem]


class UnmaskRequest(BaseModel):
    request_id: str
    items: list[TextItem] = Field(min_length=1)
    delete_context: bool = False
    allow_missing_context: bool = True


class UnmaskResultItem(BaseModel):
    id: str
    text: str
    replacements: int


class UnmaskResponse(BaseModel):
    request_id: str
    context_found: bool
    replacements: int
    context_deleted: bool
    items: list[UnmaskResultItem]


class StreamUnmaskRequest(BaseModel):
    request_id: str
    stream_id: str = "default"
    chunk: str
    final: bool = False
    delete_context: bool = False
    allow_missing_context: bool = True


class StreamUnmaskResponse(BaseModel):
    request_id: str
    stream_id: str
    context_found: bool
    output: str
    replacements: int
    buffered_chars: int
    final: bool
    context_deleted: bool


class FinalizeRequest(BaseModel):
    request_id: str


class FinalizeResponse(BaseModel):
    request_id: str
    context_deleted: bool


class PolicyListResponse(BaseModel):
    default_policy: str
    policies: list[str]
