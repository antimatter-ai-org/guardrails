from __future__ import annotations

import argparse
import random
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import load_policy_config
from app.core.analysis.service import PresidioAnalysisService
from app.eval_v3.config import load_eval_registry
from app.eval_v3.datasets.hf_span_dataset import build_samples_from_hf_split, load_hf_split
from app.eval_v3.metrics.spans import filter_scored_spans
from app.eval_v3.predictors.analyze_text import as_eval_spans
from app.eval_v3.tasks.span_detection import SpanDetectionInputs, run_span_detection
from app.model_assets import apply_model_env
from app.settings import settings


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit false-positive spans for a dataset/policy.")
    p.add_argument("--dataset", action="append", required=True)
    p.add_argument("--split", default="fast")
    p.add_argument("--registry-path", default=str(Path("configs") / "eval" / "suites.yaml"))
    p.add_argument("--cache-dir", default=str(Path(".eval_cache") / "hf"))
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    p.add_argument("--policy-path", default="configs/policy.yaml")
    p.add_argument("--policy-name", default="external_nemotron_only")
    p.add_argument("--max-examples-per-label", type=int, default=8)
    p.add_argument("--context-window", type=int, default=40)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--output", required=True)
    return p.parse_args()


def _ensure_runtime_ready(*, service: PresidioAnalysisService, analyzer_profile: str, policy_name: str) -> None:
    errors = service.ensure_profile_runtimes_ready(
        profile_names=[analyzer_profile],
        timeout_s=settings.pytriton_init_timeout_s,
    )
    if errors:
        raise RuntimeError(f"model runtime readiness check failed for policy '{policy_name}': {errors}")


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return min(a_end, b_end) > max(a_start, b_start)


def _mask_span(text: str, start: int, end: int, window: int) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:start] + "<PRED>" + text[end:right]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _load_samples(*, registry_path: str, dataset_id: str, split: str, cache_dir: str, hf_token_env: str):
    registry = load_eval_registry(Path(registry_path))
    cfg = registry.datasets[dataset_id]
    hf_token = None
    env_name = str(hf_token_env)
    if env_name:
        from os import getenv
        hf_token = getenv(env_name) or None
    ds, fingerprint = load_hf_split(hf_id=cfg.hf_id, split=split, cache_dir=cache_dir, hf_token=hf_token)
    samples = build_samples_from_hf_split(
        dataset_id=dataset_id,
        split=split,
        ds=ds,
        text_field=cfg.text_field,
        spans_field=cfg.spans_field,
        label_map=cfg.label_map,
        slice_fields=cfg.slice_fields,
        selected_indices=None,
        max_samples=None,
    )
    return cfg, samples, fingerprint


def main() -> None:
    args = _parse_args()
    apply_model_env(settings.model_dir, settings.offline_mode)
    policy_cfg = load_policy_config(Path(args.policy_path))
    if args.policy_name not in policy_cfg.policies:
        raise RuntimeError(f"unknown policy: {args.policy_name}")
    policy = policy_cfg.policies[args.policy_name]

    service = PresidioAnalysisService(policy_cfg)
    _ensure_runtime_ready(service=service, analyzer_profile=policy.analyzer_profile, policy_name=args.policy_name)

    output_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "policy_name": args.policy_name,
        "datasets": [],
    }

    for dataset_id in args.dataset:
        cfg, samples, fingerprint = _load_samples(
            registry_path=args.registry_path,
            dataset_id=dataset_id,
            split=args.split,
            cache_dir=args.cache_dir,
            hf_token_env=args.hf_token_env,
        )
        if args.max_samples is not None and len(samples) > int(args.max_samples):
            rng = random.Random(int(args.sample_seed))
            indices = list(range(len(samples)))
            rng.shuffle(indices)
            selected = {idx for idx in indices[: int(args.max_samples)]}
            samples = [sample for idx, sample in enumerate(samples) if idx in selected]

        inputs = [SpanDetectionInputs(dataset_id=dataset_id, split=args.split, samples=samples, scored_labels=cfg.scored_labels)]
        _, predictions_by_dataset = run_span_detection(
            service=service,
            analyzer_profile=policy.analyzer_profile,
            min_score=float(policy.min_score),
            inputs=inputs,
            num_workers=int(args.workers),
            errors_preview_limit=0,
            progress_every_samples=500,
            progress_every_seconds=20.0,
        )

        pred_by_id = predictions_by_dataset.get(dataset_id, {})
        view = filter_scored_spans(samples=samples, predictions_by_id=pred_by_id, scored_labels=cfg.scored_labels)

        fp_examples: dict[str, list[dict[str, Any]]] = {}
        fp_counts: dict[str, int] = {}
        total_fp = 0

        for sample in view.samples:
            gold_by_label: dict[str, list[tuple[int, int]]] = {}
            for sp in sample.gold_spans:
                if not sp.canonical_label:
                    continue
                gold_by_label.setdefault(sp.canonical_label, []).append((int(sp.start), int(sp.end)))

            preds = view.predictions_by_id.get(sample.sample_id, [])
            for pred in preds:
                label = pred.canonical_label
                if not label:
                    continue
                gold_ranges = gold_by_label.get(label, [])
                matched = any(_overlaps(int(pred.start), int(pred.end), gs, ge) for gs, ge in gold_ranges)
                if matched:
                    continue
                fp_counts[label] = fp_counts.get(label, 0) + 1
                total_fp += 1

                if label not in fp_examples:
                    fp_examples[label] = []
                if len(fp_examples[label]) >= int(args.max_examples_per_label):
                    continue

                span_text = sample.text[int(pred.start) : int(pred.end)]
                fp_examples[label].append(
                    {
                        "sample_id": sample.sample_id,
                        "span_start": int(pred.start),
                        "span_end": int(pred.end),
                        "span_len": int(pred.end) - int(pred.start),
                        "span_hash": _hash_text(span_text),
                        "context": _mask_span(sample.text, int(pred.start), int(pred.end), int(args.context_window)),
                        "format": sample.metadata.get("format"),
                        "length_bucket": sample.metadata.get("length_bucket"),
                        "source": sample.metadata.get("source"),
                    }
                )

        dataset_payload = {
            "dataset_id": dataset_id,
            "split": args.split,
            "dataset_fingerprint": fingerprint,
            "sample_count": len(samples),
            "scored_labels": sorted(cfg.scored_labels),
            "false_positive_total": total_fp,
            "false_positive_by_label": dict(sorted(fp_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "examples_by_label": fp_examples,
        }
        output_payload["datasets"].append(dataset_payload)

    Path(args.output).write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
