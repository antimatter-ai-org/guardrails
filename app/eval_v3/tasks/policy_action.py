from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.eval.types import EvalSample, EvalSpan
from app.eval_v3.metrics.classification import BinaryCounts
from app.eval_v3.metrics.spans import filter_scored_spans
from app.eval_v3.reporting.schema import binary_counts_payload


@dataclass(frozen=True, slots=True)
class PolicyActionInputs:
    dataset_id: str
    split: str
    samples: list[EvalSample]
    predictions_by_id: dict[str, list[EvalSpan]]
    scored_labels: frozenset[str]


def _counts_for_inputs(items: list[PolicyActionInputs]) -> BinaryCounts:
    tp = fp = tn = fn = 0
    for item in items:
        view = filter_scored_spans(samples=item.samples, predictions_by_id=item.predictions_by_id, scored_labels=item.scored_labels)
        for sample in view.samples:
            gold_pos = len(sample.gold_spans) > 0
            pred_pos = len(view.predictions_by_id.get(sample.sample_id, [])) > 0
            if gold_pos and pred_pos:
                tp += 1
            elif (not gold_pos) and pred_pos:
                fp += 1
            elif gold_pos and (not pred_pos):
                fn += 1
            else:
                tn += 1
    return BinaryCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def run_policy_action(
    *,
    inputs_by_policy: dict[str, list[PolicyActionInputs]],
    positive_action_by_policy: dict[str, str] | None = None,
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    total_samples = 0
    for policy_name, inputs in sorted(inputs_by_policy.items()):
        counts = _counts_for_inputs(inputs)
        total_samples = max(total_samples, sum(len(item.samples) for item in inputs))
        positive_action = (positive_action_by_policy or {}).get(policy_name) or "MASKED"
        policies[policy_name] = {
            "positive_action": positive_action,
            "metrics": binary_counts_payload(counts),
        }

    return {"sample_count": total_samples, "policies": policies}
