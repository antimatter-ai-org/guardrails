from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import load_policy_config
from app.core.analysis.service import PresidioAnalysisService
from app.eval.env import load_env_file
from app.eval_v3.config import EvalRegistry, load_eval_registry
from app.eval_v3.datasets.hf_span_dataset import build_samples_from_hf_split, load_hf_split, scan_hf_span_dataset
from app.eval_v3.datasets.splits import SamplerSpec, SubsetSpec, load_or_create_indices
from app.eval_v3.reporting.schema import REPORT_VERSION
from app.eval_v3.reporting.write_outputs import write_report_files
from app.eval_v3.tasks.leakage import MaskLeakageInputs, run_mask_leakage
from app.eval_v3.tasks.policy_action import PolicyActionInputs, run_policy_action
from app.eval_v3.tasks.span_detection import SpanDetectionInputs, run_span_detection
from app.eval_v3.predictors.analyze_text import as_eval_spans
from app.model_assets import apply_model_env
from app.settings import settings


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Guardrails evaluation runner (v3).")
    p.add_argument("--registry-path", default=str(Path("configs") / "eval" / "suites.yaml"))
    p.add_argument("--suite", default="guardrails_ru")
    p.add_argument("--dataset", action="append", default=None, help="Dataset id override (repeatable).")
    p.add_argument("--split", default=None, help="Split override (fast|full). Default uses suite default.")
    p.add_argument("--tasks", default="all", help="Tasks: all|span_detection,policy_action,mask_leakage")

    p.add_argument("--policy-path", default="configs/policy.yaml")
    p.add_argument("--policy-name", default=None, help="Policy used for span_detection/mask_leakage (default: policy.yaml default_policy).")
    p.add_argument("--action-policies", default="external_default,strict_block", help="Comma-separated policy ids for policy_action.")

    p.add_argument("--cache-dir", default=".eval_cache/hf", help="HF hub/datasets cache root (unless HF_HOME already set).")
    p.add_argument("--output-dir", default="reports/evaluations")
    p.add_argument("--env-file", default=".env.eval")
    p.add_argument("--hf-token-env", default="HF_TOKEN")

    p.add_argument("--offline", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--runtime", choices=["cpu", "cuda"], default=None)
    p.add_argument("--cpu-device", choices=["auto", "cpu", "mps"], default=None)

    p.add_argument("--subset", default="all", help="Subset: all|negatives|positives|language=..|script_profile=..")
    p.add_argument("--sampler", choices=["none", "random", "label_balanced"], default="none")
    p.add_argument("--sampler-size", type=int, default=None)
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--min-per-label", type=int, default=None, help="For sampler=label_balanced.")

    p.add_argument("--max-samples", type=int, default=None, help="Cap samples per dataset after subset/sampling.")
    p.add_argument("--errors-preview-limit", type=int, default=25)
    p.add_argument("--progress-every-samples", type=int, default=1000)
    p.add_argument("--progress-every-seconds", type=float, default=15.0)
    p.add_argument("--workers", type=int, default=1, help="Span detection parallelism (ThreadPool). Default: 1.")
    return p.parse_args()


def _configure_hf_cache(cache_dir: str, *, force: bool) -> None:
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    if force or "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(cache_path)
    if force or "HUGGINGFACE_HUB_CACHE" not in os.environ:
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "hub")
    if force or "HF_DATASETS_CACHE" not in os.environ:
        os.environ["HF_DATASETS_CACHE"] = str(cache_path / "datasets")


def _configure_offline(offline: bool) -> None:
    if not offline:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _resolve_suite_and_datasets(registry: EvalRegistry, suite_name: str, dataset_overrides: list[str] | None) -> tuple[str, list[str], str]:
    suite = registry.suites.get(suite_name)
    if suite is None:
        raise ValueError(f"unknown suite: {suite_name} (known: {sorted(registry.suites)})")
    datasets = list(dataset_overrides) if dataset_overrides else list(suite.datasets)
    for ds in datasets:
        if ds not in registry.datasets:
            raise ValueError(f"unknown dataset id in selection: {ds}")
    return suite.name, datasets, suite.default_split


def _load_samples_for_dataset(
    *,
    registry: EvalRegistry,
    dataset_id: str,
    split: str,
    cache_dir: str,
    hf_token: str | bool | None,
    subset: SubsetSpec,
    sampler: SamplerSpec,
    max_samples: int | None,
) -> tuple[list[Any], SpanDetectionInputs, PolicyActionInputs, MaskLeakageInputs, dict[str, Any]]:
    cfg = registry.datasets[dataset_id]
    if cfg.kind != "hf_span_v1":
        raise ValueError(f"unsupported dataset kind for {dataset_id}: {cfg.kind}")

    ds, fingerprint = load_hf_split(hf_id=cfg.hf_id, split=split, cache_dir=cache_dir, hf_token=hf_token)
    default_subset = subset.raw.strip() in {"", "all"}
    default_sampler = sampler.name == "none"
    if default_subset and default_sampler:
        selection = None
        samples = build_samples_from_hf_split(
            dataset_id=dataset_id,
            split=split,
            ds=ds,
            text_field=cfg.text_field,
            spans_field=cfg.spans_field,
            label_map=cfg.label_map,
            slice_fields=cfg.slice_fields,
            selected_indices=None,
            max_samples=max_samples,
        )
    else:
        entity_counts, languages, script_profiles, label_sets = scan_hf_span_dataset(ds=ds, spans_field=cfg.spans_field, label_map=cfg.label_map)
        selection = load_or_create_indices(
            cache_dir=cache_dir,
            dataset_id=dataset_id,
            split=split,
            subset=subset,
            sampler=sampler,
            dataset_fingerprint=fingerprint,
            entity_counts=entity_counts,
            languages=languages,
            script_profiles=script_profiles,
            sample_label_sets=label_sets,
        )
        samples = build_samples_from_hf_split(
            dataset_id=dataset_id,
            split=split,
            ds=ds,
            text_field=cfg.text_field,
            spans_field=cfg.spans_field,
            label_map=cfg.label_map,
            slice_fields=cfg.slice_fields,
            selected_indices=selection.indices,
            max_samples=max_samples,
        )

    meta = {
        "dataset_id": dataset_id,
        "hf_id": cfg.hf_id,
        "split": split,
        "sample_count": len(samples),
        "dataset_fingerprint": fingerprint,
        "indices_cache_path": str(selection.cache_path) if selection is not None else None,
        "indices_from_cache": bool(selection.from_cache) if selection is not None else False,
        "subset": subset.raw,
        "sampler": {"name": sampler.name, "seed": sampler.seed, "size": sampler.size, "min_per_label": sampler.min_per_label},
    }

    return (
        samples,
        SpanDetectionInputs(dataset_id=dataset_id, split=split, samples=samples, scored_labels=cfg.scored_labels),
        PolicyActionInputs(dataset_id=dataset_id, split=split, samples=samples, predictions_by_id={}, scored_labels=cfg.scored_labels),
        MaskLeakageInputs(dataset_id=dataset_id, split=split, samples=samples, predictions_by_id={}, scored_labels=cfg.scored_labels),
        meta,
    )


def _ensure_runtime_ready(
    *,
    service: PresidioAnalysisService,
    analyzer_profile: str,
    policy_name: str,
) -> None:
    errors = service.ensure_profile_runtimes_ready(profile_names=[analyzer_profile], timeout_s=settings.pytriton_init_timeout_s)
    if errors:
        raise RuntimeError(f"model runtime readiness check failed for policy '{policy_name}': {errors}")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _maybe_start_embedded_pytriton() -> Any | None:
    if settings.runtime_mode != "cuda":
        return None
    from app.runtime.pytriton_embedded import EmbeddedPyTritonConfig, EmbeddedPyTritonManager

    manager = EmbeddedPyTritonManager(
        EmbeddedPyTritonConfig(
            pytriton_url=settings.pytriton_url,
            gliner_model_ref=_env("GR_PYTRITON_GLINER_MODEL_REF", "urchade/gliner_multi-v2.1"),
            token_model_ref=_env("GR_PYTRITON_TOKEN_MODEL_REF", "scanpatch/pii-ner-nemotron"),
            model_dir=settings.model_dir,
            offline_mode=settings.offline_mode,
            device=_env("GR_PYTRITON_DEVICE", "cuda"),
            max_batch_size=int(_env("GR_PYTRITON_MAX_BATCH_SIZE", "32")),
            enable_nemotron=settings.enable_nemotron,
            grpc_port=int(_env("GR_PYTRITON_GRPC_PORT", "8001")),
            metrics_port=int(_env("GR_PYTRITON_METRICS_PORT", "8002")),
            readiness_timeout_s=settings.pytriton_init_timeout_s,
        )
    )
    manager.start()
    settings.pytriton_url = manager.client_url
    return manager


def main() -> int:
    args = _parse_args()
    load_env_file(args.env_file)
    run_id = f"evalv3_{args.suite}_{args.split or 'default'}_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}"
    started_at = datetime.now(tz=UTC)
    run_started = time.perf_counter()

    registry = load_eval_registry(args.registry_path)
    suite_name, dataset_ids, suite_default_split = _resolve_suite_and_datasets(registry, args.suite, args.dataset)
    split = str(args.split or suite_default_split)
    # Now that split is resolved, make the run id stable/accurate.
    run_id = f"evalv3_{suite_name}_{split}_{started_at.strftime('%Y%m%dT%H%M%SZ')}"

    tasks_raw = str(args.tasks).strip()
    if tasks_raw == "all":
        tasks = ["span_detection", "policy_action", "mask_leakage"]
    else:
        tasks = [item.strip() for item in tasks_raw.split(",") if item.strip()]

    # Cache configuration: if user passes non-default cache_dir, force override.
    force_cache = str(args.cache_dir) != ".eval_cache/hf"
    _configure_hf_cache(args.cache_dir, force=force_cache)
    _configure_offline(bool(args.offline) or (os.getenv("GR_OFFLINE_MODE", "").lower() in {"1", "true", "yes", "on"}))

    # If the user didn't override --cache-dir but HF_HOME is already set (common in remote runs),
    # prefer the existing HF_HOME for both env + datasets.load_dataset(cache_dir=...).
    effective_cache_dir = str(args.cache_dir)
    if not force_cache:
        hf_home = os.getenv("HF_HOME")
        if hf_home:
            effective_cache_dir = str(Path(hf_home).expanduser().resolve())

    if args.runtime is not None:
        os.environ["GR_RUNTIME_MODE"] = str(args.runtime)
        settings.runtime_mode = str(args.runtime)  # align already-imported settings with CLI flags
    if args.cpu_device is not None:
        os.environ["GR_CPU_DEVICE"] = str(args.cpu_device)
        settings.cpu_device = str(args.cpu_device)

    hf_token_env = os.getenv(args.hf_token_env)
    offline_effective = bool(args.offline) or (os.getenv("HF_HUB_OFFLINE") == "1") or (os.getenv("HF_DATASETS_OFFLINE") == "1")
    # If HF_TOKEN isn't set, fall back to the locally cached HF auth token (hf auth login),
    # which is enabled by passing token=True to datasets/huggingface_hub calls.
    hf_token: str | bool | None = hf_token_env if hf_token_env else (True if not offline_effective else None)

    # Policy selection for span_detection / leakage.
    policy_cfg = load_policy_config(args.policy_path)
    span_policy_name = args.policy_name or policy_cfg.default_policy
    if span_policy_name not in policy_cfg.policies:
        raise RuntimeError(f"unknown policy '{span_policy_name}'")
    span_policy = policy_cfg.policies[span_policy_name]

    # Model cache/offline env is handled by existing runtime code.
    apply_model_env(
        model_dir=os.getenv("GR_MODEL_DIR"),
        offline_mode=os.getenv("GR_OFFLINE_MODE", "").lower() in {"1", "true", "yes", "on"},
    )

    manager = _maybe_start_embedded_pytriton()

    subset = SubsetSpec(raw=str(args.subset))
    sampler = SamplerSpec(
        name=str(args.sampler),
        seed=int(args.sampler_seed),
        size=int(args.sampler_size) if args.sampler_size is not None else None,
        min_per_label=int(args.min_per_label) if args.min_per_label is not None else None,
    )

    # Load datasets once (HF download/cache will be reused by datasets library).
    dataset_load_meta: list[dict[str, Any]] = []
    span_inputs: list[SpanDetectionInputs] = []
    policy_inputs: list[PolicyActionInputs] = []
    leakage_inputs: list[MaskLeakageInputs] = []
    samples_by_dataset: dict[str, list[Any]] = {}

    dataset_load_started = time.perf_counter()
    for dataset_id in dataset_ids:
        dataset_started = time.perf_counter()
        samples, span_in, pol_in, leak_in, meta = _load_samples_for_dataset(
            registry=registry,
            dataset_id=dataset_id,
            split=split,
            cache_dir=effective_cache_dir,
            hf_token=hf_token,
            subset=subset,
            sampler=sampler,
            max_samples=args.max_samples,
        )
        meta["load_elapsed_seconds"] = round(time.perf_counter() - dataset_started, 6)
        samples_by_dataset[dataset_id] = samples
        span_inputs.append(span_in)
        policy_inputs.append(pol_in)
        leakage_inputs.append(leak_in)
        dataset_load_meta.append(meta)
        print(
            f"[loaded] dataset={dataset_id} split={split} samples={meta['sample_count']} indices_cache={meta['indices_cache_path']} cached={meta['indices_from_cache']}",
            flush=True,
            file=sys.stderr,
        )
    dataset_load_elapsed = time.perf_counter() - dataset_load_started

    def predict_eval_spans_for_policy(
        *,
        service: PresidioAnalysisService,
        analyzer_profile: str,
        min_score: float,
    ) -> dict[str, dict[str, list[Any]]]:
        predictions_by_dataset: dict[str, dict[str, list[Any]]] = {}
        for ds in span_inputs:
            pred_by_id: dict[str, list[Any]] = {}
            for sample in ds.samples:
                detections = service.analyze_text(text=sample.text, profile_name=analyzer_profile, policy_min_score=min_score)
                pred_by_id[sample.sample_id] = as_eval_spans(detections)
            predictions_by_dataset[ds.dataset_id] = pred_by_id
        return predictions_by_dataset

    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run": {
            "run_id": run_id,
            "started_at_utc": started_at.isoformat(),
            "suite": suite_name,
            "split": split,
            "datasets": dataset_ids,
            "tasks": tasks,
            "registry_path": str(Path(args.registry_path).resolve()),
            "policy_path": str(Path(args.policy_path).resolve()),
            "policy_name": span_policy_name,
            "subset": subset.raw,
            "sampler": {"name": sampler.name, "seed": sampler.seed, "size": sampler.size, "min_per_label": sampler.min_per_label},
            "max_samples": args.max_samples,
            "cache_dir": str(Path(effective_cache_dir).resolve()),
            "offline": bool(args.offline),
            "runtime": {
                "mode": os.getenv("GR_RUNTIME_MODE", "cpu"),
                "cpu_device": os.getenv("GR_CPU_DEVICE", "auto"),
            },
            "settings": {
                "enable_nemotron": bool(settings.enable_nemotron),
                "pytriton_url": str(settings.pytriton_url),
            },
            "env": {
                # Keep this minimal: enough to reproduce runtime decisions, but avoid leaking
                # hostnames/connection strings into committed baselines/docs.
                "GR_RUNTIME_MODE": os.getenv("GR_RUNTIME_MODE"),
                "GR_CPU_DEVICE": os.getenv("GR_CPU_DEVICE"),
                "GR_ENABLE_NEMOTRON": os.getenv("GR_ENABLE_NEMOTRON"),
                "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            },
            "dataset_load": dataset_load_meta,
            "timing": {
                "dataset_load_seconds": round(dataset_load_elapsed, 6),
            },
        },
        "tasks": {},
    }

    service = PresidioAnalysisService(policy_cfg)

    # Span detection (single policy) + export EvalSpan predictions for downstream tasks.
    span_predictions_by_dataset: dict[str, dict[str, list[Any]]] | None = None
    try:
        if "span_detection" in tasks or "mask_leakage" in tasks:
            span_started = time.perf_counter()
            _ensure_runtime_ready(service=service, analyzer_profile=span_policy.analyzer_profile, policy_name=span_policy_name)
            span_report, span_predictions_by_dataset = run_span_detection(
                service=service,
                analyzer_profile=span_policy.analyzer_profile,
                min_score=float(span_policy.min_score),
                inputs=span_inputs,
                num_workers=int(args.workers),
                errors_preview_limit=int(args.errors_preview_limit),
                progress_every_samples=int(args.progress_every_samples),
                progress_every_seconds=float(args.progress_every_seconds),
            )
            report["tasks"]["span_detection"] = span_report
            report["run"]["timing"]["span_detection_seconds"] = round(time.perf_counter() - span_started, 6)

            for leak_in in leakage_inputs:
                leak_in.predictions_by_id.update(span_predictions_by_dataset.get(leak_in.dataset_id, {}))

        # Policy action: run for configured policies (policy-specific analyzer profiles).
        if "policy_action" in tasks:
            action_started = time.perf_counter()
            action_policy_names = _split_list(args.action_policies)
            inputs_by_policy: dict[str, list[PolicyActionInputs]] = {}
            positive_action_by_policy: dict[str, str] = {}

            for policy_name in action_policy_names:
                if policy_name not in policy_cfg.policies:
                    raise RuntimeError(f"unknown policy for --action-policies: {policy_name}")
                pol = policy_cfg.policies[policy_name]
                positive_action_by_policy[policy_name] = "BLOCKED" if pol.mode == "block" else "MASKED"

                # Reuse span predictions if the action policy matches the span policy.
                if span_predictions_by_dataset is not None and policy_name == span_policy_name:
                    preds = span_predictions_by_dataset
                else:
                    _ensure_runtime_ready(service=service, analyzer_profile=pol.analyzer_profile, policy_name=policy_name)
                    preds = predict_eval_spans_for_policy(
                        service=service, analyzer_profile=pol.analyzer_profile, min_score=float(pol.min_score)
                    )

                policy_specific_inputs: list[PolicyActionInputs] = []
                for base in policy_inputs:
                    policy_specific_inputs.append(
                        PolicyActionInputs(
                            dataset_id=base.dataset_id,
                            split=base.split,
                            samples=base.samples,
                            predictions_by_id=preds.get(base.dataset_id, {}),
                            scored_labels=base.scored_labels,
                        )
                    )
                inputs_by_policy[policy_name] = policy_specific_inputs

            action_report = run_policy_action(
                inputs_by_policy=inputs_by_policy,
                positive_action_by_policy=positive_action_by_policy,
            )
            report["tasks"]["policy_action"] = action_report
            report["run"]["timing"]["policy_action_seconds"] = round(time.perf_counter() - action_started, 6)

        # Leakage: uses span predictions + deterministic masking.
        if "mask_leakage" in tasks:
            leakage_started = time.perf_counter()
            if span_predictions_by_dataset is None:
                raise RuntimeError(
                    "mask_leakage requires span predictions; include span_detection or run without task filtering"
                )
            leak_report = run_mask_leakage(inputs=leakage_inputs, errors_preview_limit=int(args.errors_preview_limit))
            report["tasks"]["mask_leakage"] = leak_report
            report["run"]["timing"]["mask_leakage_seconds"] = round(time.perf_counter() - leakage_started, 6)
    finally:
        if manager is not None:
            try:
                manager.stop()
            except Exception:
                pass

    finished_at = datetime.now(tz=UTC)
    report["run"]["finished_at_utc"] = finished_at.isoformat()
    report["run"]["timing"]["wall_seconds"] = round(time.perf_counter() - run_started, 6)

    outputs = write_report_files(report=report, output_dir=args.output_dir, run_id=run_id)
    print(json.dumps(outputs, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
