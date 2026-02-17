from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalTask:
    """
    Evaluation task identifier.

    v3 of the evaluator is designed to support multiple tasks over time, e.g.:
    - pii_spans_v1: span-level PII detection (current)
    - prompt_injection_v1: classify injection attempts
    - action_block_v1: expected Guardrails action (BLOCKED/FLAGGED/NONE)
    """

    task_id: str

