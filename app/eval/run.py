from __future__ import annotations

import argparse
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
from app.eval.metrics import evaluate_samples, match_counts
from app.eval.predictor import merge_cascade_detections, sample_uncertainty_score
from app.eval.report import build_report_payload, metrics_payload, write_report_files
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
    parser.add_argument("--mode", choices=("baseline", "cascade"), default="baseline")
    parser.add_argument("--cascade-threshold", type=float, default=0.15)
    parser.add_argument(
        "--cascade-heavy-recognizers",
        default="gliner_pii_multilingual",
        help="Comma-separated recognizer ids to run in stage B for cascade mode.",
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


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_dataset_split(
    *,
    dataset_name: str,
    requested_split: str,
    hf_token: str | None,
    strict_split: bool,
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

    for sample in samples:
        source = str(sample.metadata.get("source") or "unknown")
        by_source[source].append(sample)
        noisy_val = sample.metadata.get("noisy")
        if isinstance(noisy_val, bool):
            noisy_key = "true" if noisy_val else "false"
        else:
            noisy_key = "unknown"
        by_noisy[noisy_key].append(sample)

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


def _warm_up_profile(service: PresidioAnalysisService, profile_name: str) -> None:
    profile = service._config.analyzer_profiles[profile_name]  # noqa: SLF001
    if service._requires_analyzer_engine(profile):  # noqa: SLF001
        recognizers = service._get_engine(profile_name).registry.recognizers  # noqa: SLF001
    else:
        recognizers = service._get_registry(profile_name).recognizers  # noqa: SLF001

    for recognizer in recognizers:
        runtime = getattr(recognizer, "_runtime", None)
        if runtime is None:
            continue

        if getattr(runtime, "_model", None) is None and callable(getattr(runtime, "_load_model", None)):
            runtime._load_model()  # noqa: SLF001

        detect_fn = getattr(recognizer, "analyze", None)
        if callable(detect_fn):
            try:
                detect_fn("warmup", [], None)
            except Exception:
                pass


def _make_cascade_services(
    policy_config: Any,
    policy_name: str,
    heavy_recognizers: set[str],
) -> tuple[PresidioAnalysisService, str, PresidioAnalysisService | None, str | None]:
    policy = policy_config.policies[policy_name]
    profile_name = policy.analyzer_profile
    profile = policy_config.analyzer_profiles[profile_name]
    recognizers = list(profile.analysis.recognizers)

    stage_a_recognizers = [name for name in recognizers if name not in heavy_recognizers]
    stage_b_recognizers = [name for name in recognizers if name in heavy_recognizers]

    if not stage_a_recognizers:
        stage_a_recognizers = recognizers
        stage_b_recognizers = []

    config_stage_a = policy_config.model_copy(deep=True)
    profile_a_name = f"{profile_name}__stage_a"
    config_stage_a.analyzer_profiles[profile_a_name] = config_stage_a.analyzer_profiles[profile_name].model_copy(deep=True)
    config_stage_a.analyzer_profiles[profile_a_name].analysis.recognizers = stage_a_recognizers
    service_a = PresidioAnalysisService(config_stage_a)

    if not stage_b_recognizers:
        return service_a, profile_a_name, None, None

    config_stage_b = policy_config.model_copy(deep=True)
    profile_b_name = f"{profile_name}__stage_b"
    config_stage_b.analyzer_profiles[profile_b_name] = config_stage_b.analyzer_profiles[profile_name].model_copy(deep=True)
    config_stage_b.analyzer_profiles[profile_b_name].analysis.recognizers = stage_b_recognizers
    service_b = PresidioAnalysisService(config_stage_b)
    return service_a, profile_a_name, service_b, profile_b_name


def main() -> int:
    args = _parse_args()
    load_env_file(args.env_file)
    _configure_hf_cache(args.cache_dir)

    hf_token = os.getenv(args.hf_token_env)
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    from app.settings import settings

    apply_model_env(model_dir=settings.model_dir, offline_mode=settings.offline_mode)

    policy_config = load_policy_config(args.policy_path)
    policy_name = args.policy_name or policy_config.default_policy
    if policy_name not in policy_config.policies:
        raise KeyError(f"Policy '{policy_name}' not found in {args.policy_path}")
    policy = policy_config.policies[policy_name]

    selected_datasets = args.dataset or list_supported_datasets()
    heavy_recognizers = set(_parse_csv(args.cascade_heavy_recognizers))

    if args.mode == "cascade":
        stage_a_service, stage_a_profile, stage_b_service, stage_b_profile = _make_cascade_services(
            policy_config=policy_config,
            policy_name=policy_name,
            heavy_recognizers=heavy_recognizers,
        )
        _warm_up_profile(stage_a_service, stage_a_profile)
        if stage_b_service is not None and stage_b_profile is not None:
            _warm_up_profile(stage_b_service, stage_b_profile)
    else:
        baseline_service = PresidioAnalysisService(policy_config)
        _warm_up_profile(baseline_service, policy.analyzer_profile)

    overall_started = time.perf_counter()

    dataset_reports: list[dict[str, Any]] = []
    all_samples: list[EvalSample] = []
    all_predictions_by_id: dict[str, list[EvalSpan]] = {}
    all_detector_predictions: defaultdict[str, dict[str, list[EvalSpan]]] = defaultdict(dict)
    combined_errors_preview: list[dict[str, object]] = []
    total_escalated = 0

    for dataset_name in selected_datasets:
        adapter = get_dataset_adapter(dataset_name)
        split_to_use, available_splits = _resolve_dataset_split(
            dataset_name=dataset_name,
            requested_split=args.split,
            hf_token=hf_token,
            strict_split=args.strict_split,
        )
        if split_to_use is None:
            available_text = ", ".join(available_splits) if available_splits else "unknown"
            print(
                (
                    f"[warn] dataset={dataset_name} missing split='{args.split}' "
                    f"(available: {available_text}); skipping due to --strict-split"
                ),
                flush=True,
            )
            continue
        if split_to_use != args.split:
            available_text = ", ".join(available_splits) if available_splits else "unknown"
            print(
                (
                    f"[warn] dataset={dataset_name} missing split='{args.split}', "
                    f"using split='{split_to_use}' (available: {available_text})"
                ),
                flush=True,
            )
        samples = adapter.load_samples(
            split=split_to_use,
            cache_dir=args.cache_dir,
            hf_token=hf_token,
            max_samples=args.max_samples,
        )
        if not samples:
            continue

        split_used = str(samples[0].metadata.get("__split__") or args.split)
        print(
            f"[progress] dataset={dataset_name} split={split_used} loaded_samples={len(samples)}",
            flush=True,
        )
        predictions_by_id: dict[str, list[EvalSpan]] = {}
        detector_predictions: defaultdict[str, dict[str, list[EvalSpan]]] = defaultdict(dict)
        errors_preview: list[dict[str, object]] = []
        dataset_escalated = 0

        started = time.perf_counter()
        last_progress_time = started
        last_progress_count = 0
        total_samples = len(samples)
        for idx, sample in enumerate(samples, start=1):
            if args.mode == "cascade":
                _, stage_a_detections = stage_a_service.analyze_text(
                    text=sample.text,
                    profile_name=stage_a_profile,
                    policy_min_score=policy.min_score,
                    language_hint=None,
                )
                uncertainty = sample_uncertainty_score(
                    sample.text,
                    stage_a_detections=stage_a_detections,
                    min_score=policy.min_score,
                )

                if stage_b_service is not None and stage_b_profile is not None and uncertainty >= args.cascade_threshold:
                    dataset_escalated += 1
                    _, stage_b_detections = stage_b_service.analyze_text(
                        text=sample.text,
                        profile_name=stage_b_profile,
                        policy_min_score=policy.min_score,
                        language_hint=None,
                    )
                    detections = merge_cascade_detections(stage_a_detections, stage_b_detections)
                else:
                    detections = stage_a_detections
            else:
                _, detections = baseline_service.analyze_text(
                    text=sample.text,
                    profile_name=policy.analyzer_profile,
                    policy_min_score=policy.min_score,
                    language_hint=None,
                )

            predicted_spans = _as_eval_spans(detections)
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
                            "dataset": dataset_name,
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

            now = time.perf_counter()
            processed_since_last = idx - last_progress_count
            should_print_progress = (
                idx == total_samples
                or idx == 1
                or processed_since_last >= max(1, args.progress_every_samples)
                or (now - last_progress_time) >= max(0.1, args.progress_every_seconds)
            )
            if should_print_progress:
                elapsed = max(0.0, now - started)
                rate = (idx / elapsed) if elapsed > 0 else 0.0
                remaining = max(0, total_samples - idx)
                eta_seconds = (remaining / rate) if rate > 0 else 0.0
                cascade_suffix = ""
                if args.mode == "cascade":
                    cascade_suffix = f", escalated={dataset_escalated}"
                print(
                    (
                        f"[progress] dataset={dataset_name} split={split_used} "
                        f"{idx}/{total_samples} ({(idx / total_samples) * 100:.2f}%) "
                        f"elapsed={_format_duration(elapsed)} "
                        f"rate={rate:.2f} samples/s "
                        f"eta={_format_duration(eta_seconds)}"
                        f"{cascade_suffix}"
                    ),
                    flush=True,
                )
                last_progress_time = now
                last_progress_count = idx

        elapsed_seconds = time.perf_counter() - started
        aggregate = evaluate_samples(samples, predictions_by_id)
        print(
            (
                f"[progress] dataset={dataset_name} split={split_used} complete "
                f"samples={len(samples)} elapsed={_format_duration(elapsed_seconds)} "
                f"rate={(len(samples) / elapsed_seconds) if elapsed_seconds > 0 else 0.0:.2f} samples/s"
            ),
            flush=True,
        )

        dataset_report = {
            "name": dataset_name,
            "split": split_used,
            "sample_count": len(samples),
            "elapsed_seconds": round(elapsed_seconds, 6),
            "samples_per_second": round((len(samples) / elapsed_seconds) if elapsed_seconds > 0 else 0.0, 6),
            "metrics": metrics_payload(aggregate),
            "detector_breakdown": _detector_breakdown(samples, dict(detector_predictions)),
            "dataset_slices": _slice_metrics(samples, predictions_by_id),
            "errors_preview": errors_preview,
        }
        if args.mode == "cascade":
            dataset_report["cascade"] = {
                "threshold": args.cascade_threshold,
                "escalated_samples": dataset_escalated,
                "escalated_ratio": round(dataset_escalated / max(1, len(samples)), 6),
            }
            total_escalated += dataset_escalated

        dataset_reports.append(dataset_report)

        for sample in samples:
            global_id = f"{dataset_name}::{sample.sample_id}"
            all_samples.append(
                EvalSample(
                    sample_id=global_id,
                    text=sample.text,
                    gold_spans=sample.gold_spans,
                    metadata={**sample.metadata, "dataset": dataset_name},
                )
            )
            all_predictions_by_id[global_id] = predictions_by_id.get(sample.sample_id, [])

        for detector_name, by_sample in detector_predictions.items():
            for sample_id, spans in by_sample.items():
                all_detector_predictions[detector_name][f"{dataset_name}::{sample_id}"] = spans

        for item in errors_preview:
            if len(combined_errors_preview) >= args.errors_preview_limit:
                break
            combined_errors_preview.append(item)

    if not all_samples:
        raise RuntimeError("No samples were loaded for the requested datasets")

    elapsed_total = time.perf_counter() - overall_started
    aggregate_total = evaluate_samples(all_samples, all_predictions_by_id)

    if len(dataset_reports) == 1:
        top_dataset_name = dataset_reports[0]["name"]
        top_split = dataset_reports[0]["split"]
        report_slug_dataset = _dataset_slug(top_dataset_name)
    else:
        top_dataset_name = "all"
        top_split = args.split
        report_slug_dataset = "all_datasets"

    report_payload = build_report_payload(
        dataset_name=top_dataset_name,
        split=top_split,
        sample_count=len(all_samples),
        policy_name=policy_name,
        policy_path=str(Path(args.policy_path).resolve()),
        runtime_mode=settings.runtime_mode,
        elapsed_seconds=elapsed_total,
        aggregate=aggregate_total,
        errors_preview=combined_errors_preview,
    )
    report_payload["evaluation"]["mode"] = args.mode
    report_payload["datasets"] = dataset_reports
    report_payload["detector_breakdown"] = _detector_breakdown(all_samples, dict(all_detector_predictions))
    report_payload["dataset_slices"] = _slice_metrics(all_samples, all_predictions_by_id)

    if args.mode == "cascade":
        report_payload["evaluation"]["cascade"] = {
            "threshold": args.cascade_threshold,
            "stage_a_profile": stage_a_profile,
            "stage_b_profile": stage_b_profile,
            "heavy_recognizers": sorted(heavy_recognizers),
            "escalated_samples": total_escalated,
            "escalated_ratio": round(total_escalated / max(1, len(all_samples)), 6),
        }

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    report_slug = f"eval_{report_slug_dataset}_{top_split}_{args.mode}_{timestamp}"
    json_path, md_path = write_report_files(report_payload, output_dir=args.output_dir, report_slug=report_slug)

    print(f"[progress] all-datasets complete elapsed={_format_duration(elapsed_total)}", flush=True)
    print(f"[ok] Datasets: {', '.join(selected_datasets)}")
    print(f"[ok] Samples (combined): {len(all_samples)}")
    print(f"[ok] Policy: {policy_name}")
    print(f"[ok] Mode: {args.mode}")
    if args.mode == "cascade":
        print(f"[ok] Cascade escalated: {total_escalated}/{len(all_samples)}")
    print(f"[ok] JSON report: {json_path}")
    print(f"[ok] Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
