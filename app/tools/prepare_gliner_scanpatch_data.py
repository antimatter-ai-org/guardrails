from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.eval.env import load_env_file
from app.finetune.gliner_pipeline import (
    DEFAULT_SCANPATCH_DATASET,
    build_training_bundle_from_eval_samples,
    discover_dataset_splits,
    load_eval_samples_for_splits,
    write_training_jsonl,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GLiNER fine-tuning data from Scanpatch.")
    parser.add_argument("--dataset", default=DEFAULT_SCANPATCH_DATASET)
    parser.add_argument("--cache-dir", default=".eval_cache/hf")
    parser.add_argument("--output-dir", default="reports/finetune")
    parser.add_argument("--output-name", default="scanpatch_all_splits_gliner_train.jsonl")
    parser.add_argument("--summary-name", default="scanpatch_all_splits_gliner_train_summary.json")
    parser.add_argument("--splitter-type", default="whitespace")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--env-file", default=".env.eval")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_env_file(args.env_file)
    hf_token = os.getenv(args.hf_token_env)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
        splitter_type=args.splitter_type,
    )

    jsonl_path = write_training_jsonl(bundle.training_records, output_dir / args.output_name)
    summary = {
        "dataset": args.dataset,
        "splits": splits,
        "labels": bundle.labels,
        "stats": bundle.stats,
        "training_jsonl_path": jsonl_path,
    }
    summary_path = output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] dataset: {args.dataset}")
    print(f"[ok] splits: {', '.join(splits)}")
    print(f"[ok] labels: {', '.join(bundle.labels)}")
    print(f"[ok] records: {len(bundle.training_records)}")
    print(f"[ok] train jsonl: {jsonl_path}")
    print(f"[ok] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

