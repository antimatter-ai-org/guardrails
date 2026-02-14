from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.finetune.gliner_pipeline import finetune_gliner_records, read_training_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GLiNER on prepared JSONL data.")
    parser.add_argument("--train-jsonl", required=True, help="Path to prepared JSONL with tokenized_text + ner")
    parser.add_argument("--base-model", default="urchade/gliner_multi-v2.1")
    parser.add_argument("--output-dir", default="reports/finetune/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--precision", default="auto", help="auto|fp32|fp16|bf16")
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile-model", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    records = read_training_jsonl(args.train_jsonl)

    result = finetune_gliner_records(
        records=records,
        base_model=args.base_model,
        output_root=args.output_dir,
        run_name=args.run_name,
        device=args.device,
        precision=args.precision,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        compile_model=args.compile_model,
    )

    summary_path = Path(result.output_dir) / "finetune_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_name": result.run_name,
                "base_model": result.base_model,
                "output_dir": result.output_dir,
                "checkpoint_dir": result.checkpoint_dir,
                "final_model_dir": result.final_model_dir,
                "resolved_device": result.resolved_device,
                "precision_mode": result.precision_mode,
                "started_at_utc": result.started_at_utc,
                "finished_at_utc": result.finished_at_utc,
                "epochs": result.epochs,
                "max_steps": result.max_steps,
                "train_samples": result.train_samples,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ok] run: {result.run_name}")
    print(f"[ok] base model: {result.base_model}")
    print(f"[ok] device: {result.resolved_device} ({result.precision_mode})")
    print(f"[ok] final model: {result.final_model_dir}")
    print(f"[ok] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

