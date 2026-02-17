from __future__ import annotations

import sys
import time
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
    progress_every_seconds: float = 15.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    last_progress = started
    total_gold_spans = 0
    leaked_gold_spans = 0
    leaked_examples: list[dict[str, Any]] = []
    processed_samples = 0
    total_samples = sum(len(item.samples) for item in inputs)

    for item in inputs:
        view = filter_scored_spans(samples=item.samples, predictions_by_id=item.predictions_by_id, scored_labels=item.scored_labels)
        for sample in view.samples:
            processed_samples += 1
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

            now = time.perf_counter()
            if progress_every_seconds and (now - last_progress) >= float(progress_every_seconds):
                rate = processed_samples / max(1e-6, (now - started))
                eta_s = (total_samples - processed_samples) / max(1e-6, rate)
                print(
                    f"[progress] task=mask_leakage processed={processed_samples}/{total_samples} "
                    f"rate={rate:.2f}/s eta_s={eta_s:.1f}",
                    flush=True,
                    file=sys.stderr,
                )
                last_progress = now

    leakage_fraction = 0.0 if total_gold_spans == 0 else leaked_gold_spans / total_gold_spans
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "sample_count": total_samples,
        "processed_samples": processed_samples,
        "total_gold_spans": total_gold_spans,
        "leaked_gold_spans": leaked_gold_spans,
        "leakage_fraction": round(leakage_fraction, 6),
        "leaked_examples": leaked_examples,
    }
