from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.eval.aggregate import WeightedMacroResult, weighted_macro_recall
from app.eval.cache_paths import EvalCachePaths
from app.eval.collection_resolver import resolve_collection
from app.eval.compare import compare_reports
from app.eval.datasets.privacy_mask_parquet import LoadedDataset, load_hf_dataset, rows_to_samples
from app.eval.env import load_env_file
from app.eval.env_info import environment_payload
from app.eval.gates import evaluate_gates, load_gates
from app.eval.metrics import EvaluationAggregate, evaluate_samples, match_counts, merge_aggregates
from app.eval.report_v3 import render_summary_md, write_csv_rows, write_json, write_jsonl
from app.eval.suite_loader import DatasetSpec, SuiteSpec, load_suite
from app.eval.tasks.pii_spans import PII_SPANS_V1
from app.eval.types import EvalSample, EvalSpan
from app.eval.views import ViewSpec, builtin_view, parse_where_clauses, resolve_view_indices
from app.eval.weights import load_weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Guardrails evaluation suite.")
    parser.add_argument("--suite", default="guardrails-ru", help="Suite id (configs/eval/suites/<id>.yaml) or a path.")
    parser.add_argument("--dataset", action="append", default=None, help="Dataset id to include (repeatable).")
    parser.add_argument("--tag", action="append", default=None, help="Dataset tag filter (repeatable, AND semantics).")
    parser.add_argument("--split", action="append", default=None, help="Split to evaluate (repeatable). Default is suite default.")
    parser.add_argument("--full", action="store_true", help="Convenience flag to include split=full.")

    parser.add_argument("--policy-path", default="configs/policy.yaml")
    parser.add_argument("--policy-name", default=None)

    parser.add_argument("--cache-dir", default=".eval_cache", help="Eval cache root (HF caches under <root>/hf).")
    parser.add_argument("--output-dir", default="reports/evaluations")
    parser.add_argument("--env-file", default=".env.eval")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--refresh-collection", action="store_true", help="Refresh HF collection cache.")

    parser.add_argument("--runtime-mode", choices=["cpu", "cuda"], default=None, help="Set GR_RUNTIME_MODE for this run.")
    parser.add_argument("--cpu-device", choices=["auto", "cpu", "mps"], default=None, help="Set GR_CPU_DEVICE for this run.")

    parser.add_argument("--weights-path", default="configs/eval/weights.yaml")
    parser.add_argument("--gates-path", default="configs/eval/gates.yaml")
    parser.add_argument("--enforce-gates", action="store_true", help="Enforce gates even on non-full or sampled runs.")

    parser.add_argument("--view", default=None, help="Built-in view name (e.g. negative).")
    parser.add_argument("--where", action="append", default=None, help="Filter expression (repeatable). Example: language=en")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap evaluated samples per dataset+split after filtering.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-by",
        default=None,
        help="Comma-separated stratification keys for sampling (e.g. language,script_profile,label_presence).",
    )

    parser.add_argument("--compare", default=None, help="Path to a report.json (v3) to compare against.")

    parser.add_argument("--errors-preview-limit", type=int, default=25)
    parser.add_argument("--progress-every-samples", type=int, default=1000)
    parser.add_argument("--progress-every-seconds", type=float, default=15.0)
    return parser.parse_args()


def _suite_path(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p.expanduser().resolve()
    return (Path("configs") / "eval" / "suites" / f"{arg}.yaml").resolve()


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


def _ensure_runtime_models_ready(
    *,
    service: Any,
    policy_name: str,
    analyzer_profile: str,
    timeout_s: float,
) -> None:
    readiness_errors = service.ensure_profile_runtimes_ready(
        profile_names=[analyzer_profile],
        timeout_s=timeout_s,
    )
    if readiness_errors:
        raise RuntimeError(f"model runtime readiness check failed for policy '{policy_name}': {readiness_errors}")


def _as_eval_spans(detections: list[Any]) -> list[EvalSpan]:
    spans: list[EvalSpan] = []
    for item in detections:
        metadata = getattr(item, "metadata", None) or {}
        canonical = metadata.get("canonical_label")
        if canonical is None:
            canonical = str(getattr(item, "label", "")).strip().lower() or None
        spans.append(
            EvalSpan(
                start=int(getattr(item, "start", 0)),
                end=int(getattr(item, "end", 0)),
                label=str(getattr(item, "label", "")),
                canonical_label=str(canonical).lower() if canonical else None,
                score=float(getattr(item, "score", 0.0)) if getattr(item, "score", None) is not None else None,
                detector=str(getattr(item, "detector", "")) or None,
            )
        )
    return spans


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned = sorted((s, e) for s, e in ranges if e > s)
    if not cleaned:
        return []
    out: list[tuple[int, int]] = []
    cur_s, cur_e = cleaned[0]
    for s, e in cleaned[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
            continue
        out.append((cur_s, cur_e))
        cur_s, cur_e = s, e
    out.append((cur_s, cur_e))
    return out


def _range_len(ranges: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in ranges)


def _negative_slice_metrics(
    *,
    samples: list[EvalSample],
    predictions_by_id: dict[str, list[EvalSpan]],
    allowed_labels: set[str],
) -> dict[str, Any]:
    negatives = [s for s in samples if not s.gold_spans]
    if not negatives:
        return {
            "negative_samples": 0,
            "predicted_spans_per_sample": 0.0,
            "predicted_chars_per_sample": 0.0,
            "any_prediction_rate": 0.0,
        }
    spans_total = 0
    chars_total = 0
    any_pred = 0
    for s in negatives:
        preds = [p for p in predictions_by_id.get(s.sample_id, []) if p.canonical_label in allowed_labels]
        spans_total += len(preds)
        if preds:
            any_pred += 1
        merged = _merge_ranges([(p.start, p.end) for p in preds])
        chars_total += _range_len(merged)
    n = len(negatives)
    return {
        "negative_samples": n,
        "predicted_spans_per_sample": round(spans_total / n, 6),
        "predicted_chars_per_sample": round(chars_total / n, 6),
        "any_prediction_rate": round(any_pred / n, 6),
    }


def _slice_breakdown(
    *,
    samples: list[EvalSample],
    predictions_by_id: dict[str, list[EvalSpan]],
    allowed_labels: set[str],
    fields: list[str],
) -> dict[str, Any]:
    breakdown: dict[str, Any] = {}
    for field in fields:
        groups: dict[str, list[EvalSample]] = defaultdict(list)
        for s in samples:
            val = s.metadata.get(field)
            key = str(val) if val is not None else "unknown"
            groups[key].append(s)
        payload: dict[str, Any] = {}
        for key, group in sorted(groups.items()):
            agg = evaluate_samples(group, predictions_by_id, allowed_labels=allowed_labels, include_per_label=False)
            payload[key] = {
                "sample_count": len(group),
                "overlap_canonical": {
                    k: v for k, v in (("precision", agg.overlap_canonical.precision), ("recall", agg.overlap_canonical.recall), ("f1", agg.overlap_canonical.f1))
                },
                "char_canonical": {
                    k: v for k, v in (("precision", agg.char_canonical.precision), ("recall", agg.char_canonical.recall), ("f1", agg.char_canonical.f1))
                },
            }
        breakdown[field] = payload
    return breakdown


def _detector_breakdown(
    *,
    samples: list[EvalSample],
    predictions_by_id: dict[str, list[EvalSpan]],
    allowed_labels: set[str],
) -> dict[str, Any]:
    by_detector: dict[str, dict[str, list[EvalSpan]]] = defaultdict(dict)
    for sample_id, spans in predictions_by_id.items():
        for span in spans:
            det = span.detector or "unknown"
            by_detector[det].setdefault(sample_id, []).append(span)

    payload: dict[str, Any] = {}
    for det, det_preds in sorted(by_detector.items()):
        agg = evaluate_samples(samples, det_preds, allowed_labels=allowed_labels, include_per_label=False)
        payload[det] = {
            "prediction_count": sum(len(v) for v in det_preds.values()),
            "overlap_canonical": {
                "precision": round(agg.overlap_canonical.precision, 6),
                "recall": round(agg.overlap_canonical.recall, 6),
                "f1": round(agg.overlap_canonical.f1, 6),
            },
            "char_canonical": {
                "precision": round(agg.char_canonical.precision, 6),
                "recall": round(agg.char_canonical.recall, 6),
                "f1": round(agg.char_canonical.f1, 6),
            },
        }
    return payload


def _load_and_select_datasets(
    *,
    suite: SuiteSpec,
    collection_ids: set[str] | None,
    requested_datasets: list[str] | None,
    requested_tags: list[str] | None,
) -> list[DatasetSpec]:
    ds = list(suite.datasets)
    if collection_ids is not None:
        ds = [d for d in ds if d.dataset_id in collection_ids]
    if requested_datasets:
        requested = {str(x).strip() for x in requested_datasets if str(x).strip()}
        ds = [d for d in ds if d.dataset_id in requested]
    if requested_tags:
        tags = {str(x).strip().lower() for x in requested_tags if str(x).strip()}
        ds = [d for d in ds if tags.issubset(set(d.tags))]
    return ds


def _scored_entity_count_fn(
    *,
    spec: DatasetSpec,
    scored_labels: set[str],
) -> tuple[Any, Any]:
    annotated = set(spec.annotated_labels)
    mapping = spec.gold_label_mapping
    mask_field = spec.mask_field

    from app.eval.datasets.privacy_mask_parquet import canonicalize_gold_label

    def scored_count(row: dict[str, Any]) -> int:
        items = row.get(mask_field) or []
        total = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            raw = str(it.get("label") or "").strip()
            canonical = canonicalize_gold_label(raw_label=raw, mapping=mapping)
            if canonical is None:
                continue
            if canonical not in annotated or canonical not in scored_labels:
                continue
            total += 1
        return total

    def label_presence(row: dict[str, Any]) -> str:
        items = row.get(mask_field) or []
        labels: set[str] = set()
        for it in items:
            if not isinstance(it, dict):
                continue
            raw = str(it.get("label") or "").strip()
            canonical = canonicalize_gold_label(raw_label=raw, mapping=mapping)
            if canonical is None:
                continue
            if canonical not in annotated or canonical not in scored_labels:
                continue
            labels.add(canonical)
        return ",".join(sorted(labels))

    return scored_count, label_presence


def main() -> int:
    args = _parse_args()
    load_env_file(args.env_file)
    hf_token = os.getenv(args.hf_token_env)

    if args.runtime_mode:
        os.environ["GR_RUNTIME_MODE"] = args.runtime_mode
    if args.cpu_device:
        os.environ["GR_CPU_DEVICE"] = args.cpu_device

    cache_paths = EvalCachePaths.from_cache_dir_arg(args.cache_dir)

    # Import runtime modules after env is configured so Settings() reads the right values.
    from app.config import load_policy_config
    from app.core.analysis.service import PresidioAnalysisService
    from app.model_assets import apply_model_env
    from app.settings import settings

    suite = load_suite(_suite_path(args.suite))
    weights = load_weights(args.weights_path)
    gates = load_gates(args.gates_path)

    # Optional: reconcile suite vs HF collection.
    collection_ids: set[str] | None = None
    collection_info: dict[str, Any] | None = None
    if suite.default_collection:
        col = resolve_collection(
            collection=suite.default_collection,
            cache_paths=cache_paths,
            hf_token=hf_token,
            refresh=bool(args.refresh_collection),
        )
        collection_ids = set(col.dataset_ids)
        collection_info = {
            "collection": col.collection,
            "dataset_ids": list(col.dataset_ids),
            "source": col.source,
            "cache_path": str(col.cache_path),
        }

    dataset_specs = _load_and_select_datasets(
        suite=suite,
        collection_ids=collection_ids,
        requested_datasets=args.dataset,
        requested_tags=args.tag,
    )
    if not dataset_specs:
        raise RuntimeError("no datasets selected (check --dataset/--tag filters)")

    splits = args.split[:] if args.split else [suite.default_split or "fast"]
    if args.full and "full" not in splits:
        splits.append("full")
    splits = [str(s).strip() for s in splits if str(s).strip()]
    if not splits:
        splits = ["fast"]

    config = load_policy_config(args.policy_path)
    policy_name = args.policy_name or config.default_policy
    if policy_name not in config.policies:
        raise RuntimeError(f"Unknown policy '{policy_name}'")
    policy = config.policies[policy_name]

    apply_model_env(
        model_dir=os.getenv("GR_MODEL_DIR"),
        offline_mode=os.getenv("GR_OFFLINE_MODE", "").lower() in {"1", "true", "yes", "on"},
    )
    service = PresidioAnalysisService(config)
    _ensure_runtime_models_ready(
        service=service,
        policy_name=policy_name,
        analyzer_profile=policy.analyzer_profile,
        timeout_s=settings.pytriton_init_timeout_s,
    )

    scored_labels = set(suite.scored_labels)
    progress_every_samples = max(1, int(args.progress_every_samples))
    progress_every_seconds = max(0.0, float(args.progress_every_seconds))

    where = list(parse_where_clauses(args.where))
    view_name = None
    if args.view:
        builtin_where, view_name = builtin_view(args.view)
        where = list(builtin_where) + where

    stratify_by: tuple[str, ...] = tuple()
    if args.stratify_by:
        stratify_by = tuple([x.strip() for x in str(args.stratify_by).split(",") if x.strip()])

    run_started = time.perf_counter()
    dataset_reports: list[dict[str, Any]] = []
    suite_aggregates: list[EvaluationAggregate] = []
    suite_errors: list[dict[str, Any]] = []

    for split in splits:
        for spec in dataset_specs:
            allowed_labels = set(spec.annotated_labels) & scored_labels
            if not allowed_labels:
                print(f"[skip] dataset={spec.dataset_id} split={split} no allowed_labels", flush=True)
                continue

            loaded = load_hf_dataset(dataset_id=spec.dataset_id, split=split, cache_paths=cache_paths, hf_token=hf_token)

            scored_count_fn, label_presence_fn = _scored_entity_count_fn(spec=spec, scored_labels=scored_labels)
            indices, view_meta = resolve_view_indices(
                dataset_rows=loaded.rows,
                dataset_id=loaded.dataset_id,
                dataset_fingerprint=loaded.fingerprint,
                spec=spec,
                cache_paths=cache_paths,
                view=ViewSpec(
                    base_split=split,
                    where=tuple(where),
                    max_samples=args.max_samples,
                    seed=int(args.seed),
                    stratify_by=stratify_by,
                    view_name=view_name,
                ),
                scored_entity_count_fn=scored_count_fn,
                label_presence_fn=label_presence_fn if "label_presence" in stratify_by else None,
            )

            samples = rows_to_samples(dataset=loaded, spec=spec, scored_labels=scored_labels, indices=indices)
            if not samples:
                print(f"[skip] dataset={spec.dataset_id} split={split} returned no samples", flush=True)
                continue

            dataset_started = time.perf_counter()
            predictions_by_id: dict[str, list[EvalSpan]] = {}
            last_progress_time = time.perf_counter()

            for idx, sample in enumerate(samples, start=1):
                detections = service.analyze_text(
                    text=sample.text,
                    profile_name=policy.analyzer_profile,
                    policy_min_score=policy.min_score,
                )
                spans = _as_eval_spans(detections)
                # Label-safe: ignore predictions for labels not annotated by this dataset.
                spans = [s for s in spans if s.canonical_label in allowed_labels]
                predictions_by_id[sample.sample_id] = spans

                now = time.perf_counter()
                should_print = (idx % progress_every_samples == 0) or (
                    progress_every_seconds > 0 and (now - last_progress_time) >= progress_every_seconds
                )
                if should_print:
                    elapsed = now - dataset_started
                    rate = idx / elapsed if elapsed > 0 else 0.0
                    remaining = len(samples) - idx
                    eta_s = remaining / rate if rate > 0 else 0.0
                    print(
                        f"[progress] dataset={spec.dataset_id} split={split} processed={idx}/{len(samples)} "
                        f"rate={rate:.2f}/s elapsed={_format_duration(elapsed)} eta={_format_duration(eta_s)}",
                        flush=True,
                    )
                    last_progress_time = now

            dataset_elapsed = time.perf_counter() - dataset_started

            aggregate = evaluate_samples(samples, predictions_by_id, allowed_labels=allowed_labels)
            suite_aggregates.append(aggregate)

            dataset_errors: list[dict[str, Any]] = []
            for s in samples:
                predicted = predictions_by_id.get(s.sample_id, [])
                counts = match_counts(s.gold_spans, predicted, require_label=True, allow_overlap=True)
                if counts.false_positives > 0 or counts.false_negatives > 0:
                    dataset_errors.append(
                        {
                            "dataset_id": spec.dataset_id,
                            "split": split,
                            "sample_id": s.sample_id,
                            "false_positives": counts.false_positives,
                            "false_negatives": counts.false_negatives,
                            "text": s.text,
                            "gold_spans": [asdict(s2) for s2 in s.gold_spans],
                            "predicted_spans": [asdict(p) for p in predicted],
                        }
                    )

            suite_errors.extend(dataset_errors)
            dataset_reports.append(
                {
                    "dataset_id": spec.dataset_id,
                    "split": split,
                    "available_splits": list(loaded.available_splits),
                    "fingerprint": loaded.fingerprint,
                    "sample_count": len(samples),
                    "elapsed_seconds": round(dataset_elapsed, 6),
                    "samples_per_second": round(len(samples) / dataset_elapsed, 6) if dataset_elapsed > 0 else 0.0,
                    "view": view_meta,
                    "annotated_labels": sorted(set(spec.annotated_labels)),
                    "allowed_labels": sorted(allowed_labels),
                    "tags": list(spec.tags),
                    "notes": spec.notes,
                    "metrics": {
                        "micro": {
                            "exact_canonical": {
                                "true_positives": aggregate.exact_canonical.true_positives,
                                "false_positives": aggregate.exact_canonical.false_positives,
                                "false_negatives": aggregate.exact_canonical.false_negatives,
                                "precision": round(aggregate.exact_canonical.precision, 6),
                                "recall": round(aggregate.exact_canonical.recall, 6),
                                "f1": round(aggregate.exact_canonical.f1, 6),
                                "residual_miss_ratio": round(1.0 - aggregate.exact_canonical.recall, 6),
                            },
                            "overlap_canonical": {
                                "true_positives": aggregate.overlap_canonical.true_positives,
                                "false_positives": aggregate.overlap_canonical.false_positives,
                                "false_negatives": aggregate.overlap_canonical.false_negatives,
                                "precision": round(aggregate.overlap_canonical.precision, 6),
                                "recall": round(aggregate.overlap_canonical.recall, 6),
                                "f1": round(aggregate.overlap_canonical.f1, 6),
                                "residual_miss_ratio": round(1.0 - aggregate.overlap_canonical.recall, 6),
                            },
                            "char_canonical": {
                                "true_positives": aggregate.char_canonical.true_positives,
                                "false_positives": aggregate.char_canonical.false_positives,
                                "false_negatives": aggregate.char_canonical.false_negatives,
                                "precision": round(aggregate.char_canonical.precision, 6),
                                "recall": round(aggregate.char_canonical.recall, 6),
                                "f1": round(aggregate.char_canonical.f1, 6),
                                "residual_miss_ratio": round(1.0 - aggregate.char_canonical.recall, 6),
                            },
                        },
                        "per_label_char": {
                            label: {
                                "true_positives": counts.true_positives,
                                "false_positives": counts.false_positives,
                                "false_negatives": counts.false_negatives,
                                "precision": round(counts.precision, 6),
                                "recall": round(counts.recall, 6),
                                "f1": round(counts.f1, 6),
                                "residual_miss_ratio": round(1.0 - counts.recall, 6),
                                "support_gold": counts.true_positives + counts.false_negatives,
                            }
                            for label, counts in sorted(aggregate.per_label_char.items())
                        },
                    },
                    "negative_slice": _negative_slice_metrics(
                        samples=samples,
                        predictions_by_id=predictions_by_id,
                        allowed_labels=allowed_labels,
                    ),
                    "detector_breakdown": _detector_breakdown(
                        samples=samples,
                        predictions_by_id=predictions_by_id,
                        allowed_labels=allowed_labels,
                    ),
                    "slices": _slice_breakdown(
                        samples=samples,
                        predictions_by_id=predictions_by_id,
                        allowed_labels=allowed_labels,
                        fields=[f for f in ("language", "script_profile", "noisy") if any(f in s.metadata for s in samples)],
                    ),
                    "errors_preview": dataset_errors[: int(args.errors_preview_limit)],
                }
            )

            print(
                f"[done] dataset={spec.dataset_id} split={split} samples={len(samples)} "
                f"elapsed={_format_duration(dataset_elapsed)} "
                f"overlap_f1={aggregate.overlap_canonical.f1:.4f} char_f1={aggregate.char_canonical.f1:.4f}",
                flush=True,
            )

    if not dataset_reports:
        raise RuntimeError("no datasets were evaluated (check split selection and filters)")

    combined = merge_aggregates(suite_aggregates)

    # Suite-level per-label aggregation: sum counts across datasets that annotate each label.
    per_label_char: dict[str, Any] = {}
    merged_per_label_char = combined.per_label_char
    for label, counts in sorted(merged_per_label_char.items()):
        per_label_char[label] = {
            "weight": float(weights.get(label, 1.0)),
            "char_canonical": {
                "true_positives": counts.true_positives,
                "false_positives": counts.false_positives,
                "false_negatives": counts.false_negatives,
                "precision": round(counts.precision, 6),
                "recall": round(counts.recall, 6),
                "f1": round(counts.f1, 6),
                "residual_miss_ratio": round(1.0 - counts.recall, 6),
                "support_gold": counts.true_positives + counts.false_negatives,
            },
        }

    headline: WeightedMacroResult = weighted_macro_recall(per_label_counts=merged_per_label_char, weights=weights)
    headline_residual = round(1.0 - float(headline.value), 6)

    total_elapsed = time.perf_counter() - run_started
    env = environment_payload()

    report: dict[str, Any] = {
        "report_version": "3.0",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "suite": {
            "suite_id": suite.suite_id,
            "collection": suite.default_collection,
            "task_id": PII_SPANS_V1.task_id,
            "collection_info": collection_info,
            "splits": splits,
            "datasets": [d.dataset_id for d in dataset_specs],
            "policy_name": policy_name,
            "policy_path": args.policy_path,
        },
        "environment": env,
        "label_scope": {
            "scored_labels": sorted(scored_labels),
            "weights": {k: float(v) for k, v in sorted(weights.items())},
            "datasets": {d.dataset_id: {"annotated_labels": list(d.annotated_labels)} for d in dataset_specs},
        },
        "evaluation": {
            "elapsed_seconds": round(total_elapsed, 6),
            "samples_per_second": round(sum(d["sample_count"] for d in dataset_reports) / total_elapsed, 6)
            if total_elapsed > 0
            else 0.0,
        },
        "results": {
            "suite_summary": {
                "headline": {
                    "risk_weighted_macro_char_recall": headline.value,
                    "risk_weighted_macro_char_residual_miss_ratio": headline_residual,
                    "covered_labels": list(headline.covered_labels),
                    "skipped_labels_no_gold": list(headline.skipped_labels),
                    "total_weight": headline.total_weight,
                },
                "micro": {
                    "exact_canonical": {
                        "true_positives": combined.exact_canonical.true_positives,
                        "false_positives": combined.exact_canonical.false_positives,
                        "false_negatives": combined.exact_canonical.false_negatives,
                        "precision": round(combined.exact_canonical.precision, 6),
                        "recall": round(combined.exact_canonical.recall, 6),
                        "f1": round(combined.exact_canonical.f1, 6),
                        "residual_miss_ratio": round(1.0 - combined.exact_canonical.recall, 6),
                    },
                    "overlap_canonical": {
                        "true_positives": combined.overlap_canonical.true_positives,
                        "false_positives": combined.overlap_canonical.false_positives,
                        "false_negatives": combined.overlap_canonical.false_negatives,
                        "precision": round(combined.overlap_canonical.precision, 6),
                        "recall": round(combined.overlap_canonical.recall, 6),
                        "f1": round(combined.overlap_canonical.f1, 6),
                        "residual_miss_ratio": round(1.0 - combined.overlap_canonical.recall, 6),
                    },
                    "char_canonical": {
                        "true_positives": combined.char_canonical.true_positives,
                        "false_positives": combined.char_canonical.false_positives,
                        "false_negatives": combined.char_canonical.false_negatives,
                        "precision": round(combined.char_canonical.precision, 6),
                        "recall": round(combined.char_canonical.recall, 6),
                        "f1": round(combined.char_canonical.f1, 6),
                        "residual_miss_ratio": round(1.0 - combined.char_canonical.recall, 6),
                    },
                },
            },
            "by_label": per_label_char,
            "by_dataset": dataset_reports,
        },
    }

    comparison_payload: dict[str, Any] | None = None
    if args.compare:
        comparison_payload = compare_reports(base_report_path=args.compare, new_report=report)
        report["comparison"] = {
            "base_report_path": comparison_payload["base_report_path"],
            "headline_delta": comparison_payload.get("headline_delta", {}),
            "top_regressions": comparison_payload.get("top_regressions", []),
            "top_improvements": comparison_payload.get("top_improvements", []),
        }

    # Determine whether to enforce gates automatically.
    auto_full = (
        splits == ["full"]
        and args.max_samples is None
        and not args.where
        and not args.view
        and not args.dataset
        and not args.tag
        and not args.stratify_by
    )
    enforce = bool(args.enforce_gates) or auto_full
    gate_result = evaluate_gates(gates=gates, report=report, enforce=enforce)
    report["gates"] = {"enforced": gate_result.enforced, "failures": list(gate_result.failures)}

    run_id = f"eval_{suite.suite_id}_{'-'.join(splits)}_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_root = Path(args.output_dir).expanduser().resolve()
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Machine-readable outputs.
    write_json(run_dir / "report.json", report)
    if comparison_payload:
        write_json(run_dir / "comparison.json", comparison_payload)

    resolved = {
        "args": vars(args),
        "suite": {
            "suite_id": suite.suite_id,
            "default_collection": suite.default_collection,
            "default_split": suite.default_split,
            "scored_labels": list(suite.scored_labels),
            "datasets": [d.dataset_id for d in suite.datasets],
        },
        "selected_datasets": [d.dataset_id for d in dataset_specs],
        "weights_path": str(Path(args.weights_path).expanduser().resolve()),
        "gates_path": str(Path(args.gates_path).expanduser().resolve()),
        "collection": collection_info,
        "cache": {
            "root": str(cache_paths.root),
            "hf": str(cache_paths.hf),
            "views": str(cache_paths.views),
            "collections": str(cache_paths.collections),
        },
    }
    write_json(run_dir / "config.resolved.json", resolved)

    # Human report.
    (run_dir / "summary.md").write_text(render_summary_md(report), encoding="utf-8")

    # Metrics CSV (flat, parse-friendly).
    csv_rows: list[dict[str, Any]] = []
    head = report["results"]["suite_summary"]["headline"]
    csv_rows.append(
        {
            "scope": "suite",
            "metric": "risk_weighted_macro_char_recall",
            "value": head["risk_weighted_macro_char_recall"],
        }
    )
    csv_rows.append(
        {
            "scope": "suite",
            "metric": "risk_weighted_macro_char_residual_miss_ratio",
            "value": head["risk_weighted_macro_char_residual_miss_ratio"],
        }
    )
    for fam in ("exact_canonical", "overlap_canonical", "char_canonical"):
        m = report["results"]["suite_summary"]["micro"][fam]
        csv_rows.append({"scope": "suite", "metric": f"{fam}.precision", "value": m["precision"]})
        csv_rows.append({"scope": "suite", "metric": f"{fam}.recall", "value": m["recall"]})
        csv_rows.append({"scope": "suite", "metric": f"{fam}.f1", "value": m["f1"]})

    for label, item in sorted(report["results"]["by_label"].items()):
        m = item["char_canonical"]
        csv_rows.append(
            {
                "scope": "label",
                "label": label,
                "metric": "char_canonical.recall",
                "value": m["recall"],
                "support_gold": m["support_gold"],
                "weight": item.get("weight", 1.0),
            }
        )

    for ds in report["results"]["by_dataset"]:
        m = ds["metrics"]["micro"]["overlap_canonical"]
        csv_rows.append(
            {
                "scope": "dataset",
                "dataset_id": ds["dataset_id"],
                "split": ds["split"],
                "metric": "overlap_canonical.recall",
                "value": m["recall"],
                "samples": ds["sample_count"],
            }
        )

    write_csv_rows(run_dir / "metrics.csv", csv_rows)

    # Full mismatch examples.
    write_jsonl(run_dir / "errors.jsonl", suite_errors)

    print(json.dumps({"run_dir": str(run_dir)}, ensure_ascii=False))

    if gate_result.enforced and gate_result.failures:
        raise RuntimeError("evaluation gates failed: " + "; ".join(gate_result.failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
