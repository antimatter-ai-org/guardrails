from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import load_policy_config
from app.eval.datasets.registry import get_dataset_adapter
from app.eval.env import load_env_file
from app.eval.labels import canonicalize_prediction_label
from app.eval.metrics import evaluate_samples, match_counts
from app.eval.predictor import (
    detect_only,
    detect_with_policy,
    merge_cascade_detections,
    sample_uncertainty_score,
)
from app.eval.report import build_report_payload, write_report_files
from app.eval.types import EvalSpan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual guardrails evaluation on public datasets.")
    parser.add_argument("--dataset", default="scanpatch/pii-ner-corpus-synthetic-controlled")
    parser.add_argument("--split", default="test")
    parser.add_argument("--policy-path", default="configs/policy.yaml")
    parser.add_argument("--policy-name", default=None)
    parser.add_argument("--cache-dir", default=".eval_cache/hf")
    parser.add_argument("--output-dir", default="reports/evaluations")
    parser.add_argument("--env-file", default=".env.eval")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--errors-preview-limit", type=int, default=25)
    parser.add_argument("--mode", choices=("baseline", "cascade"), default="baseline")
    parser.add_argument("--cascade-threshold", type=float, default=0.15)
    parser.add_argument(
        "--cascade-heavy-detectors",
        default="gliner_pii_multilingual",
        help="Comma-separated detector names to run in stage B for cascade mode.",
    )
    return parser.parse_args()


def _configure_hf_cache(cache_dir: str) -> None:
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_path / "datasets"))


def _dataset_slug(name: str) -> str:
    safe = []
    for char in name.lower():
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def _metric_dict(metric: Any) -> dict[str, float | int]:
    return {
        "true_positives": int(metric.true_positives),
        "false_positives": int(metric.false_positives),
        "false_negatives": int(metric.false_negatives),
        "precision": round(float(metric.precision), 6),
        "recall": round(float(metric.recall), 6),
        "f1": round(float(metric.f1), 6),
    }


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _detector_stage_split(
    detectors: list[object],
    heavy_names: set[str],
) -> tuple[list[object], list[object]]:
    stage_a: list[object] = []
    stage_b: list[object] = []
    for detector in detectors:
        detector_name = str(getattr(detector, "name", ""))
        if detector_name in heavy_names:
            stage_b.append(detector)
        else:
            stage_a.append(detector)

    if not stage_a and stage_b:
        # Keep at least one stage-A pass for stability.
        stage_a = list(stage_b)
        stage_b = []
    return stage_a, stage_b


def _slice_metrics(
    samples: list[Any],
    predictions_by_id: dict[str, list[EvalSpan]],
) -> dict[str, Any]:
    by_source: defaultdict[str, list[Any]] = defaultdict(list)
    by_noisy: defaultdict[str, list[Any]] = defaultdict(list)

    for sample in samples:
        source = str(sample.metadata.get("source") or "unknown")
        by_source[source].append(sample)
        noisy_val = sample.metadata.get("noisy")
        if isinstance(noisy_val, bool):
            noisy_key = "true" if noisy_val else "false"
        else:
            noisy_key = "unknown"
        by_noisy[noisy_key].append(sample)

    def evaluate_groups(groups: dict[str, list[Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, grouped_samples in sorted(groups.items()):
            aggregate = evaluate_samples(grouped_samples, predictions_by_id)
            payload[key] = {
                "sample_count": len(grouped_samples),
                "overlap_canonical": _metric_dict(aggregate.overlap_canonical),
                "exact_canonical": _metric_dict(aggregate.exact_canonical),
            }
        return payload

    return {
        "source": evaluate_groups(dict(by_source)),
        "noisy": evaluate_groups(dict(by_noisy)),
    }


def _detector_breakdown(
    samples: list[Any],
    detector_predictions: dict[str, dict[str, list[EvalSpan]]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for detector_name, predictions in sorted(detector_predictions.items()):
        aggregate = evaluate_samples(samples, predictions)
        prediction_count = sum(len(items) for items in predictions.values())
        canonical_prediction_count = sum(
            1 for items in predictions.values() for item in items if item.canonical_label is not None
        )
        payload[detector_name] = {
            "prediction_count": prediction_count,
            "canonical_prediction_count": canonical_prediction_count,
            "overlap_agnostic": _metric_dict(aggregate.overlap_agnostic),
            "overlap_canonical": _metric_dict(aggregate.overlap_canonical),
        }
    return payload


def _warm_up_detectors(detectors: list[object], timeout_seconds: float = 120.0) -> None:
    for detector in detectors:
        runtime = getattr(detector, "_runtime", None)
        if runtime is None:
            detect_fn = getattr(detector, "detect", None)
            if callable(detect_fn):
                detect_fn("warmup")
            continue

        # For local runtime we explicitly force a synchronous load to avoid
        # evaluating with missing GLiNER predictions during async warm-up.
        if getattr(runtime, "_model", None) is None and callable(getattr(runtime, "_load_model", None)):
            runtime._load_model()
            if getattr(runtime, "_model", None) is None:
                load_error = getattr(runtime, "_load_error", None)
                if load_error:
                    raise RuntimeError(f"detector warm-up failed: {load_error}")

        detect_fn = getattr(detector, "detect", None)
        if callable(detect_fn):
            detect_fn("warmup")

        loading_started = bool(getattr(runtime, "_loading_started", False))
        if not loading_started and getattr(runtime, "_model", None) is not None:
            continue

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if getattr(runtime, "_model", None) is not None:
                break
            load_error = getattr(runtime, "_load_error", None)
            if load_error:
                raise RuntimeError(f"detector warm-up failed: {load_error}")
            time.sleep(0.2)


def main() -> int:
    args = _parse_args()
    load_env_file(args.env_file)
    _configure_hf_cache(args.cache_dir)

    hf_token = os.getenv(args.hf_token_env)
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    # Load policy and detectors only after env file is applied.
    from app.detectors.factory import build_registry
    from app.settings import settings

    policy_config = load_policy_config(args.policy_path)
    policy_name = args.policy_name or policy_config.default_policy
    if policy_name not in policy_config.policies:
        raise KeyError(f"Policy '{policy_name}' not found in {args.policy_path}")
    policy = policy_config.policies[policy_name]
    heavy_detector_names = set(_parse_csv(args.cascade_heavy_detectors))

    registry = build_registry(policy_config.detector_definitions)
    detectors: list[object] = []
    for detector_name in policy.detectors:
        detector = registry.get(detector_name)
        if detector is not None:
            detectors.append(detector)

    if not detectors:
        raise RuntimeError("No active detectors available for evaluation policy")
    _warm_up_detectors(detectors)
    stage_a_detectors, stage_b_detectors = _detector_stage_split(detectors, heavy_detector_names)

    adapter = get_dataset_adapter(args.dataset)
    samples = adapter.load_samples(
        split=args.split,
        cache_dir=args.cache_dir,
        hf_token=hf_token,
        max_samples=args.max_samples,
    )

    predictions_by_id: dict[str, list[EvalSpan]] = {}
    detector_predictions: defaultdict[str, dict[str, list[EvalSpan]]] = defaultdict(dict)
    errors_preview: list[dict[str, object]] = []
    cascade_escalated = 0

    started = time.perf_counter()
    for sample in samples:
        if args.mode == "baseline":
            detections = detect_with_policy(sample.text, policy=policy, detectors=detectors)
        else:
            stage_a_findings = detect_only(sample.text, policy=policy, detectors=stage_a_detectors)
            uncertainty = sample_uncertainty_score(
                sample.text,
                stage_a_detections=stage_a_findings,
                min_score=policy.min_score,
            )
            if stage_b_detectors and uncertainty >= args.cascade_threshold:
                cascade_escalated += 1
                stage_b_findings = detect_only(sample.text, policy=policy, detectors=stage_b_detectors)
                detections = merge_cascade_detections(stage_a_findings, stage_b_findings)
            else:
                detections = stage_a_findings

        predicted_spans = [
            EvalSpan(
                start=item.start,
                end=item.end,
                label=item.label,
                canonical_label=canonicalize_prediction_label(item.label),
                score=item.score,
                detector=item.detector,
            )
            for item in detections
        ]
        predictions_by_id[sample.sample_id] = predicted_spans
        for span in predicted_spans:
            item_detector = span.detector or "unknown"
            detector_predictions[item_detector].setdefault(sample.sample_id, []).append(span)

        if len(errors_preview) < args.errors_preview_limit:
            counts = match_counts(
                gold_spans=sample.gold_spans,
                predicted_spans=predicted_spans,
                require_label=False,
                allow_overlap=False,
            )
            if counts.false_positives > 0 or counts.false_negatives > 0:
                errors_preview.append(
                    {
                        "sample_id": sample.sample_id,
                        "text": sample.text[:500],
                        "gold_spans": [
                            {"start": span.start, "end": span.end, "label": span.label}
                            for span in sample.gold_spans
                        ],
                        "predicted_spans": [
                            {"start": span.start, "end": span.end, "label": span.label}
                            for span in predicted_spans
                        ],
                    }
                )

    elapsed_seconds = time.perf_counter() - started
    aggregate = evaluate_samples(samples, predictions_by_id)

    report_payload = build_report_payload(
        dataset_name=args.dataset,
        split=args.split,
        sample_count=len(samples),
        policy_name=policy_name,
        policy_path=str(Path(args.policy_path).resolve()),
        runtime_mode=settings.runtime_mode,
        elapsed_seconds=elapsed_seconds,
        aggregate=aggregate,
        errors_preview=errors_preview,
    )
    report_payload["evaluation"]["mode"] = args.mode
    if args.mode == "cascade":
        report_payload["evaluation"]["cascade"] = {
            "threshold": args.cascade_threshold,
            "stage_a_detectors": [str(getattr(detector, "name", "")) for detector in stage_a_detectors],
            "stage_b_detectors": [str(getattr(detector, "name", "")) for detector in stage_b_detectors],
            "escalated_samples": cascade_escalated,
            "escalated_ratio": round(cascade_escalated / max(1, len(samples)), 6),
        }
    report_payload["detector_breakdown"] = _detector_breakdown(samples, dict(detector_predictions))
    report_payload["dataset_slices"] = _slice_metrics(samples, predictions_by_id)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    report_slug = f"eval_{_dataset_slug(args.dataset)}_{args.split}_{args.mode}_{timestamp}"
    json_path, md_path = write_report_files(report_payload, output_dir=args.output_dir, report_slug=report_slug)

    print(f"[ok] Dataset: {args.dataset} ({args.split})")
    print(f"[ok] Samples: {len(samples)}")
    print(f"[ok] Policy: {policy_name}")
    print(f"[ok] Mode: {args.mode}")
    if args.mode == "cascade":
        print(f"[ok] Cascade escalated: {cascade_escalated}/{len(samples)}")
    print(f"[ok] JSON report: {json_path}")
    print(f"[ok] Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
