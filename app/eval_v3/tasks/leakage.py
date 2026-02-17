from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.eval.types import EvalSample, EvalSpan
from app.eval_v3.metrics.spans import filter_scored_spans
from app.eval_v3.predictors.masking import mask_text_with_detections
from app.models.entities import Detection


@dataclass(frozen=True, slots=True)
class MaskLeakageInputs:
    dataset_id: str
    split: str
    samples: list[EvalSample]
    predictions_by_id: dict[str, list[EvalSpan]]
    scored_labels: frozenset[str]
    placeholder_prefix: str = "GR00"


def _eval_spans_to_detections(*, text: str, spans: list[EvalSpan]) -> list[Detection]:
    detections: list[Detection] = []
    for span in spans:
        if span.canonical_label is None:
            continue
        start = int(span.start)
        end = int(span.end)
        if end <= start or start < 0 or end > len(text):
            continue
        detections.append(
            Detection(
                start=start,
                end=end,
                text=text[start:end],
                label=str(span.label),
                score=float(span.score or 1.0),
                detector=str(span.detector or "eval_v3"),
                metadata={"canonical_label": str(span.canonical_label)},
            )
        )
    return detections


def run_mask_leakage(
    *,
    inputs: list[MaskLeakageInputs],
    errors_preview_limit: int = 25,
) -> dict[str, Any]:
    total_gold_spans = 0
    leaked_gold_spans = 0
    leaked_examples: list[dict[str, Any]] = []

    for item in inputs:
        view = filter_scored_spans(samples=item.samples, predictions_by_id=item.predictions_by_id, scored_labels=item.scored_labels)
        for sample in view.samples:
            if not sample.gold_spans:
                continue
            pred_spans = view.predictions_by_id.get(sample.sample_id, [])
            detections = _eval_spans_to_detections(text=sample.text, spans=pred_spans)
            masked = mask_text_with_detections(
                text=sample.text,
                detections=detections,
                placeholder_prefix=item.placeholder_prefix,
            )
            masked_text = masked.masked_text

            for gold in sample.gold_spans:
                if gold.canonical_label is None:
                    continue
                gold_value = sample.text[gold.start : gold.end]
                total_gold_spans += 1
                if gold_value and (gold_value in masked_text):
                    leaked_gold_spans += 1
                    if len(leaked_examples) < int(errors_preview_limit):
                        leaked_examples.append(
                            {
                                "dataset_id": item.dataset_id,
                                "sample_id": sample.sample_id,
                                "label": gold.canonical_label,
                                "gold_value": gold_value,
                                "masked_preview": masked_text[:500],
                            }
                        )

    leakage_fraction = 0.0 if total_gold_spans == 0 else leaked_gold_spans / total_gold_spans
    return {
        "total_gold_spans": total_gold_spans,
        "leaked_gold_spans": leaked_gold_spans,
        "leakage_fraction": round(leakage_fraction, 6),
        "leaked_examples": leaked_examples,
    }

