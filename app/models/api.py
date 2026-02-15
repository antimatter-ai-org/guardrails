from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal["INPUT", "OUTPUT", "TOOL_INPUT", "TOOL_OUTPUT", "RETRIEVAL"]
ActionType = Literal["NONE", "MASKED", "BLOCKED", "FLAGGED"]
OutputScope = Literal["INTERVENTIONS", "FULL"]
TraceLevel = Literal["NONE", "BASIC", "FULL"]
SeverityType = Literal["low", "medium", "high", "critical"]
TransformType = Literal["reversible_mask"]
TransformMode = Literal["DEIDENTIFY", "REIDENTIFY"]


class ContentItem(BaseModel):
    id: str
    text: str


class SessionConfig(BaseModel):
    id: str | None = None
    ttl_seconds: int | None = Field(default=None, ge=1)
    allow_missing_context: bool | None = None


class ReversibleMaskTransform(BaseModel):
    type: TransformType = "reversible_mask"
    mode: TransformMode
    session: SessionConfig = Field(default_factory=SessionConfig)


class ApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None
    source: SourceType
    content: list[ContentItem] = Field(min_length=1)
    transforms: list[ReversibleMaskTransform] = Field(default_factory=list)
    output_scope: OutputScope = "FULL"
    trace: TraceLevel = "NONE"


class StreamChunk(BaseModel):
    id: str = "default"
    chunk: str
    final: bool = False


class ApplyStreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None
    source: SourceType = "OUTPUT"
    transforms: list[ReversibleMaskTransform] = Field(default_factory=list)
    output_scope: OutputScope = "FULL"
    trace: TraceLevel = "NONE"
    stream: StreamChunk


class FindingSpan(BaseModel):
    start: int
    end: int
    snippet: str
    label: str


class Finding(BaseModel):
    check_id: str
    category: str
    severity: SeverityType
    confidence: float
    spans: list[FindingSpan] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class OutputItem(BaseModel):
    id: str
    text: str


class SessionState(BaseModel):
    id: str
    ttl_seconds: int
    expires_at: str | None = None


class UsageInfo(BaseModel):
    input_items: int
    input_chars: int
    output_items: int
    output_chars: int


class TimingInfo(BaseModel):
    total_ms: float
    detector_timing_ms: dict[str, float] = Field(default_factory=dict)


class ApplyResponse(BaseModel):
    action: ActionType
    source: SourceType
    policy_id: str
    policy_version: str | None = None
    outputs: list[OutputItem] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    session: SessionState | None = None
    usage: UsageInfo
    timings: TimingInfo


class ApplyStreamResponse(BaseModel):
    action: ActionType
    source: SourceType
    policy_id: str
    policy_version: str | None = None
    stream: StreamChunk
    output_chunk: str
    replacements: int
    buffered_chars: int
    findings: list[Finding] = Field(default_factory=list)
    session: SessionState | None = None
    usage: UsageInfo
    timings: TimingInfo


class SessionFinalizeResponse(BaseModel):
    session_id: str
    context_deleted: bool


class CapabilitiesResponse(BaseModel):
    service: str
    api_version: str
    sources: list[SourceType]
    actions: list[ActionType]
    transforms: list[TransformType]
    transform_modes: list[TransformMode]
    output_scopes: list[OutputScope]
    trace_levels: list[TraceLevel]
    policies: list[str]
    checks: list[str]
    runtime_mode: Literal["cpu", "cuda"]
