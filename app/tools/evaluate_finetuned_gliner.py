from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.eval.env import load_env_file
from app.finetune.gliner_pipeline import (
    DEFAULT_SCANPATCH_DATASET,
    build_training_bundle_from_eval_samples,
    discover_dataset_splits,
    evaluate_model_on_eval_samples,
    load_eval_samples_for_splits,
)


def _metric_dict(metric: Any) -> dict[str, int | float]:
    return {
        "true_positives": int(metric.true_positives),
        "false_positives": int(metric.false_positives),
        "false_negatives": int(metric.false_negatives),
        "precision": round(float(metric.precision), 6),
        "recall": round(float(metric.recall), 6),
        "f1": round(float(metric.f1), 6),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned GLiNER checkpoint on Scanpatch-style datasets.")
    parser.add_argument("--model-ref", required=True, help="HF model id or local path to a GLiNER checkpoint.")
    parser.add_argument("--dataset", default=DEFAULT_SCANPATCH_DATASET)
    parser.add_argument("--cache-dir", default=".eval_cache/hf")
    parser.add_argument("--output-dir", default="reports/finetune/eval")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--flat-ner", action="store_true")
    parser.add_argument("--batch-size-eval", type=int, default=24)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--skip-overlap-metrics", action="store_true")
    parser.add_argument("--skip-per-label-metrics", action="store_true")
    parser.add_argument("--env-file", default=".env.eval")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_env_file(args.env_file)
    hf_token = os.getenv(args.hf_token_env)

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    splits = discover_dataset_splits(args.dataset, cache_dir=args.cache_dir, hf_token=hf_token)
    samples = load_eval_samples_for_splits(
        dataset_name=args.dataset,
        splits=splits,
        cache_dir=args.cache_dir,
        hf_token=hf_token,
        max_samples_per_split=args.max_samples_per_split,
    )
    bundle = build_training_bundle_from_eval_samples(
        dataset_name=args.dataset,
        splits=splits,
        samples=samples,
        splitter_type="whitespace",
    )

    eval_result = evaluate_model_on_eval_samples(
        model_ref=args.model_ref,
        samples=bundle.samples,
        labels=bundle.labels,
        threshold=args.threshold,
        flat_ner=args.flat_ner,
        batch_size=args.batch_size_eval,
        device=args.device,
        include_overlap=not args.skip_overlap_metrics,
        include_per_label=not args.skip_per_label_metrics,
    )

    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_root / f"gliner_eval_report_{stamp}.json"
    md_path = output_root / f"gliner_eval_report_{stamp}.md"

    report = {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "model_ref": str(args.model_ref),
        "dataset": {
            "name": bundle.dataset_name,
            "splits": bundle.splits,
            "sample_count": len(bundle.samples),
            "labels": bundle.labels,
        },
        "config": {
            "threshold": args.threshold,
            "flat_ner": args.flat_ner,
            "batch_size_eval": args.batch_size_eval,
            "device": args.device,
            "skip_overlap_metrics": args.skip_overlap_metrics,
            "skip_per_label_metrics": args.skip_per_label_metrics,
        },
        "metrics": {
            "exact_canonical": _metric_dict(eval_result.aggregate.exact_canonical),
            "overlap_canonical": None
            if args.skip_overlap_metrics
            else _metric_dict(eval_result.aggregate.overlap_canonical),
            "exact_agnostic": _metric_dict(eval_result.aggregate.exact_agnostic),
            "overlap_agnostic": None
            if args.skip_overlap_metrics
            else _metric_dict(eval_result.aggregate.overlap_agnostic),
            "per_label_exact": None
            if args.skip_per_label_metrics
            else {label: _metric_dict(metric) for label, metric in sorted(eval_result.aggregate.per_label_exact.items())},
        },
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# GLiNER Evaluation Report",
        "",
        f"- Model: `{args.model_ref}`",
        f"- Dataset: `{bundle.dataset_name}` ({', '.join(bundle.splits)})",
        f"- Samples: `{len(bundle.samples)}`",
        f"- Exact canonical F1: `{report['metrics']['exact_canonical']['f1']}`",
        f"- Exact canonical precision: `{report['metrics']['exact_canonical']['precision']}`",
        f"- Exact canonical recall: `{report['metrics']['exact_canonical']['recall']}`",
        "",
        "## Config",
        "",
        f"- Threshold: `{args.threshold}`",
        f"- Flat NER: `{args.flat_ner}`",
        f"- Skip overlap metrics: `{args.skip_overlap_metrics}`",
        f"- Skip per-label metrics: `{args.skip_per_label_metrics}`",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[ok] report json: {json_path}")
    print(f"[ok] report md: {md_path}")
    print(f"[ok] exact canonical f1: {report['metrics']['exact_canonical']['f1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
