from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import load_policy_config
from app.core.analysis.service import PresidioAnalysisService
from app.eval.datasets.registry import get_dataset_adapter, list_supported_datasets
from app.eval.env import load_env_file
from app.eval.labels import canonicalize_prediction_label
from app.eval.metrics import EvaluationAggregate, evaluate_samples, match_counts
from app.eval.report import build_report_payload, metrics_payload, write_report_files
from app.eval.script_profile import classify_script_profile
from app.eval.types import EvalSample, EvalSpan
from app.model_assets import apply_model_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual guardrails evaluation on public datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Dataset name (repeatable). If omitted, all supported datasets are evaluated "
            f"({', '.join(list_supported_datasets())})."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--policy-path", default="configs/policy.yaml")
    parser.add_argument("--policy-name", default=None)
    parser.add_argument("--cache-dir", default=".eval_cache/hf")
    parser.add_argument("--output-dir", default="reports/evaluations")
    parser.add_argument("--env-file", default=".env.eval")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument(
        "--strict-split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip datasets missing the requested split instead of falling back to another split.",
    )
    parser.add_argument(
        "--synthetic-test-size",
        type=float,
        default=0.2,
        help="Test split ratio for datasets without native test split.",
    )
    parser.add_argument(
        "--synthetic-split-seed",
        type=int,
        default=42,
        help="Random seed for cached synthetic split generation.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--errors-preview-limit", type=int, default=25)
    parser.add_argument(
        "--progress-every-samples",
        type=int,
        default=1000,
        help="Emit progress every N processed samples per dataset.",
    )
    parser.add_argument(
        "--progress-every-seconds",
        type=float,
        default=15.0,
        help="Emit progress every N seconds per dataset.",
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


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _resolve_dataset_split(
    *,
    dataset_name: str,
    requested_split: str,
    hf_token: str | None,
    strict_split: bool,
    allow_synthetic_split: bool = False,
) -> tuple[str | None, list[str]]:
    try:
        from datasets import get_dataset_split_names
    except Exception:
        return requested_split, []

    try:
        split_names = [str(item) for item in get_dataset_split_names(dataset_name, token=hf_token)]
    except Exception:
        return requested_split, []

    available = set(split_names)
    if requested_split in available:
        return requested_split, sorted(available)
    if allow_synthetic_split and requested_split in {"train", "test"} and "train" in available:
        return requested_split, sorted(available)
    if strict_split:
        return None, sorted(available)
    if "train" in available:
        return "train", sorted(available)
    if available:
        return sorted(available)[0], sorted(available)
    return None, []


def _slice_metrics(
    samples: list[EvalSample],
    predictions_by_id: dict[str, list[EvalSpan]],
) -> dict[str, Any]:
    by_source: defaultdict[str, list[EvalSample]] = defaultdict(list)
    by_noisy: defaultdict[str, list[EvalSample]] = defaultdict(list)
    by_script_profile: defaultdict[str, list[EvalSample]] = defaultdict(list)

    for sample in samples:
        source = str(sample.metadata.get("source") or "unknown")
        by_source[source].append(sample)
        noisy_val = sample.metadata.get("noisy")
        if isinstance(noisy_val, bool):
            noisy_key = "true" if noisy_val else "false"
        else:
            noisy_key = "unknown"
        by_noisy[noisy_key].append(sample)
        by_script_profile[classify_script_profile(sample.text)].append(sample)

    def evaluate_groups(groups: dict[str, list[EvalSample]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, grouped_samples in sorted(groups.items()):
            aggregate = evaluate_samples(grouped_samples, predictions_by_id)
            payload[key] = {
                "sample_count": len(grouped_samples),
                "overlap_canonical": metrics_payload(aggregate)["overlap_canonical"],
                "exact_canonical": metrics_payload(aggregate)["exact_canonical"],
            }
        return payload

    return {
        "source": evaluate_groups(dict(by_source)),
        "noisy": evaluate_groups(dict(by_noisy)),
        "script_profile": evaluate_groups(dict(by_script_profile)),
    }


def _detector_breakdown(
    samples: list[EvalSample],
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
            "overlap_agnostic": metrics_payload(aggregate)["overlap_agnostic"],
            "overlap_canonical": metrics_payload(aggregate)["overlap_canonical"],
        }
    return payload


def _as_eval_spans(detections: list[Any]) -> list[EvalSpan]:
    spans: list[EvalSpan] = []
    for item in detections:
        metadata = item.metadata or {}
        canonical = metadata.get("canonical_label")
        if canonical is None:
            canonical = canonicalize_prediction_label(item.label)
        spans.append(
            EvalSpan(
                start=item.start,
                end=item.end,
                label=item.label,
                canonical_label=canonical,
                score=item.score,
                detector=item.detector,
            )
        )
    return spans


def main() -> int:
    args = _parse_args()
    load_env_file(args.env_file)
    hf_token = os.getenv(args.hf_token_env)
    _configure_hf_cache(args.cache_dir)

    config = load_policy_config(args.policy_path)
    policy_name = args.policy_name or config.default_policy
    if policy_name not in config.policies:
        raise RuntimeError(f"Unknown policy '{policy_name}'")
    policy = config.policies[policy_name]

    apply_model_env(model_dir=os.getenv("GR_MODEL_DIR"), offline_mode=os.getenv("GR_OFFLINE_MODE", "").lower() in {"1", "true", "yes", "on"})

    service = PresidioAnalysisService(config)
    dataset_names = args.dataset or list_supported_datasets()

    run_started = time.perf_counter()
    combined_samples: list[EvalSample] = []
    combined_predictions: dict[str, list[EvalSpan]] = {}
    all_errors: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []
    combined_detector_predictions: dict[str, dict[str, list[EvalSpan]]] = defaultdict(dict)

    for dataset_name in dataset_names:
        adapter = get_dataset_adapter(dataset_name)
        resolved_split, available_splits = _resolve_dataset_split(
            dataset_name=dataset_name,
            requested_split=args.split,
            hf_token=hf_token,
            strict_split=args.strict_split,
            allow_synthetic_split=adapter.supports_synthetic_split,
        )
        if resolved_split is None:
            print(
                f"[skip] dataset={dataset_name} split={args.split} unavailable "
                f"(available={','.join(available_splits) or 'unknown'})",
                flush=True,
            )
            continue

        dataset_samples = adapter.load_samples(
            split=resolved_split,
            cache_dir=args.cache_dir,
            hf_token=hf_token,
            synthetic_test_size=args.synthetic_test_size,
            synthetic_split_seed=args.synthetic_split_seed,
            max_samples=args.max_samples,
        )
        if not dataset_samples:
            print(f"[skip] dataset={dataset_name} returned no samples", flush=True)
            continue

        dataset_started = time.perf_counter()
        dataset_predictions: dict[str, list[EvalSpan]] = {}
        detector_predictions: dict[str, dict[str, list[EvalSpan]]] = defaultdict(dict)

        progress_every_samples = max(1, int(args.progress_every_samples))
        progress_every_seconds = max(0.0, float(args.progress_every_seconds))
        last_progress_time = time.perf_counter()

        for idx, sample in enumerate(dataset_samples, start=1):
            detections = service.analyze_text(
                text=sample.text,
                profile_name=policy.analyzer_profile,
                policy_min_score=policy.min_score,
            )
            spans = _as_eval_spans(detections)
            dataset_predictions[sample.sample_id] = spans

            for span in spans:
                detector_name = span.detector or "unknown"
                detector_predictions[detector_name].setdefault(sample.sample_id, []).append(span)
                combined_detector_predictions[detector_name].setdefault(sample.sample_id, []).append(span)

            now = time.perf_counter()
            should_print = (idx % progress_every_samples == 0) or (
                progress_every_seconds > 0 and (now - last_progress_time) >= progress_every_seconds
            )
            if should_print:
                elapsed = now - dataset_started
                rate = idx / elapsed if elapsed > 0 else 0.0
                remaining = len(dataset_samples) - idx
                eta_s = remaining / rate if rate > 0 else 0.0
                print(
                    f"[progress] dataset={dataset_name} split={resolved_split} processed={idx}/{len(dataset_samples)} "
                    f"rate={rate:.2f}/s elapsed={_format_duration(elapsed)} eta={_format_duration(eta_s)}",
                    flush=True,
                )
                last_progress_time = now

        dataset_elapsed = time.perf_counter() - dataset_started
        dataset_aggregate = evaluate_samples(dataset_samples, dataset_predictions)

        dataset_errors: list[dict[str, Any]] = []
        for sample in dataset_samples:
            predicted = dataset_predictions.get(sample.sample_id, [])
            counts = match_counts(sample.gold_spans, predicted, require_label=True, allow_overlap=True)
            if counts.false_positives > 0 or counts.false_negatives > 0:
                dataset_errors.append(
                    {
                        "sample_id": sample.sample_id,
                        "false_positives": counts.false_positives,
                        "false_negatives": counts.false_negatives,
                        "text": sample.text,
                    }
                )

        dataset_report = {
            "name": dataset_name,
            "split": resolved_split,
            "available_splits": available_splits,
            "sample_count": len(dataset_samples),
            "elapsed_seconds": round(dataset_elapsed, 6),
            "samples_per_second": round(len(dataset_samples) / dataset_elapsed, 6) if dataset_elapsed > 0 else 0.0,
            "metrics": metrics_payload(dataset_aggregate),
            "errors_preview": dataset_errors[: args.errors_preview_limit],
            "dataset_slices": _slice_metrics(dataset_samples, dataset_predictions),
            "detector_breakdown": _detector_breakdown(dataset_samples, detector_predictions),
        }
        dataset_reports.append(dataset_report)

        for sample in dataset_samples:
            combined_samples.append(sample)
            combined_predictions[sample.sample_id] = dataset_predictions.get(sample.sample_id, [])
        all_errors.extend(dataset_errors)

        print(
            f"[done] dataset={dataset_name} split={resolved_split} samples={len(dataset_samples)} "
            f"elapsed={_format_duration(dataset_elapsed)} "
            f"exact_f1={dataset_report['metrics']['exact_canonical']['f1']:.4f} "
            f"overlap_f1={dataset_report['metrics']['overlap_canonical']['f1']:.4f}",
            flush=True,
        )

    if not combined_samples:
        raise RuntimeError("No datasets were evaluated. Check --dataset/--split arguments.")

    total_elapsed = time.perf_counter() - run_started
    combined_aggregate = evaluate_samples(combined_samples, combined_predictions)
    combined_detector_breakdown = _detector_breakdown(combined_samples, combined_detector_predictions)

    report_payload = build_report_payload(
        dataset_name="all" if len(dataset_reports) > 1 else dataset_reports[0]["name"],
        split=args.split,
        sample_count=len(combined_samples),
        policy_name=policy_name,
        policy_path=args.policy_path,
        runtime_mode=os.getenv("GR_RUNTIME_MODE", "cpu"),
        elapsed_seconds=total_elapsed,
        aggregate=combined_aggregate,
        errors_preview=all_errors[: args.errors_preview_limit],
    )
    report_payload["datasets"] = dataset_reports
    report_payload["dataset_slices"] = _slice_metrics(combined_samples, combined_predictions)
    report_payload["detector_breakdown"] = combined_detector_breakdown
    report_payload["evaluation"]["mode"] = "baseline"
    report_payload["evaluation"]["generated_at_utc"] = datetime.now(tz=UTC).isoformat()

    report_slug = f"eval_{_dataset_slug(report_payload['dataset']['name'])}_{args.split}_baseline_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}"
    json_path, md_path = write_report_files(report_payload, args.output_dir, report_slug)

    print(json.dumps({"report_json": json_path, "report_md": md_path}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
