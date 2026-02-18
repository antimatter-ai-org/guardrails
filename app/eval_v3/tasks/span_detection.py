from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
import sys
from typing import Any

from app.core.analysis.service import PresidioAnalysisService
from app.eval.metrics import EvaluationAggregate, evaluate_samples, match_counts
from app.eval.types import EvalSample, EvalSpan
from app.eval_v3.metrics.aggregation import macro_over_labels
from app.eval_v3.metrics.risk_weighting import risk_weighted_char_recall
from app.eval_v3.metrics.spans import filter_scored_spans
from app.eval_v3.predictors.analyze_text import as_eval_spans
from app.eval_v3.reporting.schema import aggregate_payload


@dataclass(frozen=True, slots=True)
class SpanDetectionInputs:
    dataset_id: str
    split: str
    samples: list[EvalSample]
    scored_labels: frozenset[str]


def _slice_metrics(
    *,
    samples: list[EvalSample],
    predictions_by_id: dict[str, list[EvalSpan]],
    key: str,
) -> dict[str, Any]:
    groups: dict[str, list[EvalSample]] = defaultdict(list)
    for sample in samples:
        val = sample.metadata.get(key)
        groups[str(val) if val is not None else "unknown"].append(sample)

    out: dict[str, Any] = {}
    for group_key, group_samples in sorted(groups.items()):
        agg = evaluate_samples(group_samples, predictions_by_id)
        payload = aggregate_payload(agg)
        out[group_key] = {
            "sample_count": len(group_samples),
            "overlap_canonical": payload["overlap_canonical"],
            "char_canonical": payload["char_canonical"],
        }
    return out


def _detector_breakdown(
    *,
    samples: list[EvalSample],
    detector_predictions: dict[str, dict[str, list[EvalSpan]]],
    scored_labels: frozenset[str],
    scored_labels_by_dataset: dict[str, frozenset[str]] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for detector, pred_by_id in sorted(detector_predictions.items()):
        # For multi-dataset runs, enforce per-sample label supervision to avoid
        # penalizing datasets that do not annotate a label.
        if scored_labels_by_dataset is None:
            view = filter_scored_spans(samples=samples, predictions_by_id=pred_by_id, scored_labels=scored_labels)
            scored_pred_by_id = view.predictions_by_id
            scored_samples = view.samples
        else:
            scored_pred_by_id: dict[str, list[EvalSpan]] = {}
            unscored_counts: dict[str, int] = defaultdict(int)
            scored_samples = []
            for sample in samples:
                ds_id = str(sample.metadata.get("__dataset__") or "")
                ds_scored = scored_labels_by_dataset.get(ds_id, scored_labels)
                gold_scored = [
                    span
                    for span in sample.gold_spans
                    if span.canonical_label is not None and span.canonical_label in ds_scored
                ]
                scored_samples.append(
                    EvalSample(
                        sample_id=sample.sample_id,
                        text=sample.text,
                        gold_spans=gold_scored,
                        metadata=dict(sample.metadata),
                    )
                )
                spans = pred_by_id.get(sample.sample_id, [])
                scored: list[EvalSpan] = []
                for span in spans:
                    if span.canonical_label is None:
                        continue
                    if span.canonical_label in ds_scored:
                        scored.append(span)
                    else:
                        unscored_counts[str(span.canonical_label)] += 1
                scored_pred_by_id[sample.sample_id] = scored
            view = None

        agg = evaluate_samples(scored_samples, scored_pred_by_id)
        payload = aggregate_payload(agg)
        pred_count = sum(len(v) for v in pred_by_id.values())
        scored_pred_count = sum(len(v) for v in scored_pred_by_id.values())
        out[detector] = {
            "prediction_count": pred_count,
            "scored_prediction_count": scored_pred_count,
            "overlap_canonical": payload["overlap_canonical"],
            "char_canonical": payload["char_canonical"],
        }
    return out


def run_span_detection(
    *,
    service: PresidioAnalysisService,
    analyzer_profile: str,
    min_score: float,
    inputs: list[SpanDetectionInputs],
    num_workers: int = 1,
    errors_preview_limit: int = 25,
    progress_every_samples: int = 1000,
    progress_every_seconds: float = 15.0,
) -> tuple[dict[str, Any], dict[str, dict[str, list[EvalSpan]]]]:
    run_started = time.perf_counter()

    combined_samples: list[EvalSample] = []
    combined_predictions: dict[str, list[EvalSpan]] = {}
    combined_detector_predictions: dict[str, dict[str, list[EvalSpan]]] = defaultdict(dict)
    combined_unscored: dict[str, dict[str, int]] = {}
    dataset_reports: list[dict[str, Any]] = []

    predictions_by_dataset: dict[str, dict[str, list[EvalSpan]]] = {}
    scored_labels_by_dataset: dict[str, frozenset[str]] = {item.dataset_id: item.scored_labels for item in inputs}

    for item in inputs:
        dataset_started = time.perf_counter()
        predictions_by_id: dict[str, list[EvalSpan]] = {}
        detector_predictions: dict[str, dict[str, list[EvalSpan]]] = defaultdict(dict)

        last_progress = time.perf_counter()
        def _predict(sample: EvalSample) -> list[EvalSpan]:
            detections = service.analyze_text(
                text=sample.text,
                profile_name=analyzer_profile,
                policy_min_score=min_score,
            )
            return as_eval_spans(detections)

        # Parallelize at the sample level. This can drastically speed up GPU-backed
        # runtimes (pytriton) by allowing concurrent in-flight requests.
        if int(num_workers) > 1:
            with ThreadPoolExecutor(max_workers=int(num_workers)) as ex:
                for idx, (sample, spans) in enumerate(zip(item.samples, ex.map(_predict, item.samples)), start=1):
                    predictions_by_id[sample.sample_id] = spans

                    for span in spans:
                        detector = span.detector or "unknown"
                        detector_predictions[detector].setdefault(sample.sample_id, []).append(span)
                        combined_detector_predictions[detector].setdefault(sample.sample_id, []).append(span)

                    now = time.perf_counter()
                    should_print = (idx % max(1, int(progress_every_samples)) == 0) or (
                        float(progress_every_seconds) > 0 and (now - last_progress) >= float(progress_every_seconds)
                    )
                    if should_print:
                        elapsed = now - dataset_started
                        rate = idx / elapsed if elapsed > 0 else 0.0
                        remaining = len(item.samples) - idx
                        eta_s = remaining / rate if rate > 0 else 0.0
                        print(
                            f"[progress] task=span_detection dataset={item.dataset_id} split={item.split} "
                            f"processed={idx}/{len(item.samples)} rate={rate:.2f}/s eta_s={eta_s:.1f}",
                            flush=True,
                            file=sys.stderr,
                        )
                        last_progress = now
        else:
            for idx, sample in enumerate(item.samples, start=1):
                spans = _predict(sample)
                predictions_by_id[sample.sample_id] = spans

                for span in spans:
                    detector = span.detector or "unknown"
                    detector_predictions[detector].setdefault(sample.sample_id, []).append(span)
                    combined_detector_predictions[detector].setdefault(sample.sample_id, []).append(span)

                now = time.perf_counter()
                should_print = (idx % max(1, int(progress_every_samples)) == 0) or (
                    float(progress_every_seconds) > 0 and (now - last_progress) >= float(progress_every_seconds)
                )
                if should_print:
                    elapsed = now - dataset_started
                    rate = idx / elapsed if elapsed > 0 else 0.0
                    remaining = len(item.samples) - idx
                    eta_s = remaining / rate if rate > 0 else 0.0
                    print(
                        f"[progress] task=span_detection dataset={item.dataset_id} split={item.split} "
                        f"processed={idx}/{len(item.samples)} rate={rate:.2f}/s eta_s={eta_s:.1f}",
                        flush=True,
                        file=sys.stderr,
                    )
                    last_progress = now

        predictions_by_dataset[item.dataset_id] = predictions_by_id

        view = filter_scored_spans(samples=item.samples, predictions_by_id=predictions_by_id, scored_labels=item.scored_labels)
        combined_unscored[item.dataset_id] = view.unscored_predictions_by_label

        agg = evaluate_samples(view.samples, view.predictions_by_id)
        payload = aggregate_payload(agg)

        # Errors preview: overlap+label mismatches on scoped labels only.
        errors: list[dict[str, Any]] = []
        for sample in view.samples:
            pred = view.predictions_by_id.get(sample.sample_id, [])
            counts = match_counts(sample.gold_spans, pred, require_label=True, allow_overlap=True)
            if counts.false_positives > 0 or counts.false_negatives > 0:
                errors.append(
                    {
                        "sample_id": sample.sample_id,
                        "false_positives": counts.false_positives,
                        "false_negatives": counts.false_negatives,
                        "text": sample.text,
                    }
                )

        elapsed = time.perf_counter() - dataset_started
        dataset_reports.append(
            {
                "dataset_id": item.dataset_id,
                "split": item.split,
                "sample_count": len(item.samples),
                "elapsed_seconds": round(elapsed, 6),
                "samples_per_second": round(len(item.samples) / elapsed, 6) if elapsed > 0 else 0.0,
                "scored_labels": sorted(item.scored_labels),
                "metrics": payload,
                "macro_over_labels": {
                    "char": asdict(macro_over_labels(agg.per_label_char)),
                    "exact": asdict(macro_over_labels(agg.per_label_exact)),
                },
                "dataset_slices": {
                    "language": _slice_metrics(samples=view.samples, predictions_by_id=view.predictions_by_id, key="language"),
                    "script_profile": _slice_metrics(samples=view.samples, predictions_by_id=view.predictions_by_id, key="script_profile"),
                    # Long-context diagnostics (when present in dataset metadata).
                    "format": _slice_metrics(samples=view.samples, predictions_by_id=view.predictions_by_id, key="format"),
                    "length_bucket": _slice_metrics(samples=view.samples, predictions_by_id=view.predictions_by_id, key="length_bucket"),
                    "placement_profile": _slice_metrics(samples=view.samples, predictions_by_id=view.predictions_by_id, key="placement_profile"),
                    "noisy": _slice_metrics(samples=view.samples, predictions_by_id=view.predictions_by_id, key="noisy"),
                    "source": _slice_metrics(samples=view.samples, predictions_by_id=view.predictions_by_id, key="source"),
                    "entity_count": _slice_metrics(samples=view.samples, predictions_by_id=view.predictions_by_id, key="entity_count"),
                },
                "detector_breakdown": _detector_breakdown(
                    samples=item.samples,
                    detector_predictions=detector_predictions,
                    scored_labels=item.scored_labels,
                ),
                "unscored_predictions": view.unscored_predictions_by_label,
                "errors_preview": errors[: int(errors_preview_limit)],
            }
        )

        for sample in view.samples:
            combined_samples.append(sample)
            combined_predictions[sample.sample_id] = view.predictions_by_id.get(sample.sample_id, [])

        print(
            f"[done] task=span_detection dataset={item.dataset_id} split={item.split} samples={len(item.samples)} "
            f"char_recall={payload['char_canonical']['recall']:.4f} overlap_recall={payload['overlap_canonical']['recall']:.4f}",
            flush=True,
            file=sys.stderr,
        )

    combined_agg = evaluate_samples(combined_samples, combined_predictions)
    combined_payload = aggregate_payload(combined_agg)
    macro_char = macro_over_labels(combined_agg.per_label_char)
    macro_exact = macro_over_labels(combined_agg.per_label_exact)
    headline = risk_weighted_char_recall(per_label_char=combined_agg.per_label_char)

    report = {
        "elapsed_seconds": round(time.perf_counter() - run_started, 6),
        "sample_count": len(combined_samples),
        "headline": {
            "risk_weighted_char_recall": round(headline.value, 6),
            "labels_included": headline.labels_included,
            "total_weight": round(headline.total_weight, 6),
        },
        "metrics": {
            "combined": combined_payload,
            "datasets": dataset_reports,
        },
        "macro_over_labels": {
            "char": asdict(macro_char),
            "exact": asdict(macro_exact),
        },
        "dataset_slices": {
            "language": _slice_metrics(samples=combined_samples, predictions_by_id=combined_predictions, key="language"),
            "script_profile": _slice_metrics(samples=combined_samples, predictions_by_id=combined_predictions, key="script_profile"),
            "format": _slice_metrics(samples=combined_samples, predictions_by_id=combined_predictions, key="format"),
            "length_bucket": _slice_metrics(samples=combined_samples, predictions_by_id=combined_predictions, key="length_bucket"),
            "placement_profile": _slice_metrics(samples=combined_samples, predictions_by_id=combined_predictions, key="placement_profile"),
            "noisy": _slice_metrics(samples=combined_samples, predictions_by_id=combined_predictions, key="noisy"),
            "source": _slice_metrics(samples=combined_samples, predictions_by_id=combined_predictions, key="source"),
            "entity_count": _slice_metrics(samples=combined_samples, predictions_by_id=combined_predictions, key="entity_count"),
        },
        "detector_breakdown": _detector_breakdown(
            samples=combined_samples,
            detector_predictions=combined_detector_predictions,
            scored_labels=frozenset({label for item in inputs for label in item.scored_labels}),
            scored_labels_by_dataset=scored_labels_by_dataset,
        ),
        "unscored_predictions": combined_unscored,
    }

    return report, predictions_by_dataset
