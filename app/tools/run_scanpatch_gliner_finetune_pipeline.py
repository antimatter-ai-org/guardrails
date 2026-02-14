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
    evaluate_model_builtin_f1,
    evaluate_model_on_eval_samples,
    finetune_gliner_records,
    load_eval_samples_for_splits,
    write_training_jsonl,
)


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _metric_dict(metric: Any) -> dict[str, int | float]:
    return {
        "true_positives": int(metric.true_positives),
        "false_positives": int(metric.false_positives),
        "false_negatives": int(metric.false_negatives),
        "precision": round(float(metric.precision), 6),
        "recall": round(float(metric.recall), 6),
        "f1": round(float(metric.f1), 6),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def _guardrails_payload(
    *,
    threshold: float,
    flat_ner: bool,
    eval_result: Any,
    include_overlap: bool,
    include_per_label: bool,
) -> dict[str, Any]:
    per_label_exact: dict[str, Any] | None = None
    if include_per_label:
        per_label_exact = {
            label: _metric_dict(metric) for label, metric in sorted(eval_result.aggregate.per_label_exact.items())
        }

    return {
        "threshold": threshold,
        "flat_ner": flat_ner,
        "elapsed_seconds": round(eval_result.elapsed_seconds, 6),
        "sample_count": eval_result.sample_count,
        "metrics_config": {
            "include_overlap": include_overlap,
            "include_per_label": include_per_label,
        },
        "metrics": {
            "exact_canonical": _metric_dict(eval_result.aggregate.exact_canonical),
            "overlap_canonical": _metric_dict(eval_result.aggregate.overlap_canonical) if include_overlap else None,
            "exact_agnostic": _metric_dict(eval_result.aggregate.exact_agnostic),
            "overlap_agnostic": _metric_dict(eval_result.aggregate.overlap_agnostic) if include_overlap else None,
            "per_label_exact": per_label_exact,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end Scanpatch -> GLiNER fine-tune -> evaluate pipeline.")
    parser.add_argument("--dataset", default=DEFAULT_SCANPATCH_DATASET)
    parser.add_argument("--cache-dir", default=".eval_cache/hf")
    parser.add_argument("--output-dir", default="reports/finetune/scanpatch_pipeline")
    parser.add_argument("--base-model", default="urchade/gliner_multi-v2.1")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--precision", default="auto", help="auto|fp32|fp16|bf16")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--epoch-schedule", default="1.0,2.0")
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
    parser.add_argument("--splitter-type", default="whitespace")
    parser.add_argument("--thresholds", default="0.25,0.35,0.5")
    parser.add_argument("--flat-ner", action="store_true")
    parser.add_argument("--batch-size-eval", type=int, default=16)
    parser.add_argument(
        "--skip-overlap-metrics",
        action="store_true",
        help="Skip overlap metrics during guardrails evaluation to speed up large runs.",
    )
    parser.add_argument(
        "--skip-per-label-metrics",
        action="store_true",
        help="Skip per-label exact metrics during guardrails evaluation to speed up large runs.",
    )
    parser.add_argument("--skip-builtin-eval", action="store_true")
    parser.add_argument(
        "--eval-mode",
        choices=("builtin", "guardrails", "both"),
        default="builtin",
        help="builtin: GLiNER native eval only, guardrails: canonical span metrics only, both: run both.",
    )
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

    thresholds = _parse_csv_floats(args.thresholds)
    epoch_schedule = _parse_csv_floats(args.epoch_schedule)
    if not epoch_schedule:
        raise ValueError("epoch schedule must contain at least one value")

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

    train_jsonl_path = write_training_jsonl(bundle.training_records, output_root / "train_dataset.jsonl")
    prep_summary_path = output_root / "dataset_summary.json"
    prep_summary_path.write_text(
        json.dumps(
            {
                "dataset_name": bundle.dataset_name,
                "splits": bundle.splits,
                "labels": bundle.labels,
                "stats": bundle.stats,
                "train_jsonl_path": train_jsonl_path,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ok] prepared records: {len(bundle.training_records)}")
    print(f"[ok] labels: {', '.join(bundle.labels)}")
    print(f"[ok] dataset summary: {prep_summary_path}")

    best_exact_f1 = -1.0
    best_iteration_payload: dict[str, Any] | None = None
    current_base_model = args.base_model
    iteration_payloads: list[dict[str, Any]] = []

    for idx in range(args.iterations):
        iter_num = idx + 1
        epochs = epoch_schedule[min(idx, len(epoch_schedule) - 1)]
        run_name = f"iter_{iter_num:02d}"
        print(f"[run] iteration={iter_num} base_model={current_base_model} epochs={epochs}")

        finetune_result = finetune_gliner_records(
            records=bundle.training_records,
            base_model=current_base_model,
            output_root=str(output_root / "runs"),
            run_name=run_name,
            device=args.device,
            precision=args.precision,
            num_train_epochs=epochs,
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
        current_base_model = finetune_result.final_model_dir

        threshold_results: list[dict[str, Any]] = []
        best_for_iter: dict[str, Any] | None = None
        if args.eval_mode in {"guardrails", "both"}:
            for threshold in thresholds:
                eval_result = evaluate_model_on_eval_samples(
                    model_ref=finetune_result.final_model_dir,
                    samples=bundle.samples,
                    labels=bundle.labels,
                    threshold=threshold,
                    flat_ner=args.flat_ner,
                    batch_size=args.batch_size_eval,
                    device=args.device,
                    include_overlap=not args.skip_overlap_metrics,
                    include_per_label=not args.skip_per_label_metrics,
                )
                payload = _guardrails_payload(
                    threshold=threshold,
                    flat_ner=args.flat_ner,
                    eval_result=eval_result,
                    include_overlap=not args.skip_overlap_metrics,
                    include_per_label=not args.skip_per_label_metrics,
                )
                threshold_results.append(payload)
                if (
                    best_for_iter is None
                    or payload["metrics"]["exact_canonical"]["f1"] > best_for_iter["metrics"]["exact_canonical"]["f1"]
                ):
                    best_for_iter = payload

        builtin_f1: float | None = None
        builtin_raw: dict[str, Any] | None = None
        builtin_error: str | None = None
        if args.eval_mode in {"builtin", "both"} and not args.skip_builtin_eval:
            builtin_threshold = float(thresholds[0] if thresholds else 0.5)
            if best_for_iter is not None:
                builtin_threshold = float(best_for_iter["threshold"])
            try:
                builtin_f1, builtin_raw = evaluate_model_builtin_f1(
                    model_ref=finetune_result.final_model_dir,
                    records=bundle.training_records,
                    threshold=builtin_threshold,
                    flat_ner=args.flat_ner,
                    batch_size=args.batch_size_eval,
                    device=args.device,
                )
            except Exception as exc:
                builtin_error = f"{type(exc).__name__}: {exc}"
                print(f"[warn] builtin evaluation failed: {builtin_error}")
                if best_for_iter is None:
                    print("[warn] falling back to guardrails evaluation for scoring")
                    eval_result = evaluate_model_on_eval_samples(
                        model_ref=finetune_result.final_model_dir,
                        samples=bundle.samples,
                        labels=bundle.labels,
                        threshold=builtin_threshold,
                        flat_ner=args.flat_ner,
                        batch_size=args.batch_size_eval,
                        device=args.device,
                        include_overlap=not args.skip_overlap_metrics,
                        include_per_label=not args.skip_per_label_metrics,
                    )
                    fallback_payload = _guardrails_payload(
                        threshold=builtin_threshold,
                        flat_ner=args.flat_ner,
                        eval_result=eval_result,
                        include_overlap=not args.skip_overlap_metrics,
                        include_per_label=not args.skip_per_label_metrics,
                    )
                    threshold_results.append(fallback_payload)
                    best_for_iter = fallback_payload

        iter_payload = {
            "iteration": iter_num,
            "epochs": epochs,
            "finetune": {
                "run_name": finetune_result.run_name,
                "base_model": finetune_result.base_model,
                "output_dir": finetune_result.output_dir,
                "checkpoint_dir": finetune_result.checkpoint_dir,
                "final_model_dir": finetune_result.final_model_dir,
                "resolved_device": finetune_result.resolved_device,
                "precision_mode": finetune_result.precision_mode,
                "started_at_utc": finetune_result.started_at_utc,
                "finished_at_utc": finetune_result.finished_at_utc,
                "train_samples": finetune_result.train_samples,
            },
            "threshold_search": threshold_results,
            "best_threshold_result": best_for_iter,
            "builtin_eval": {
                "f1": round(float(builtin_f1), 6) if builtin_f1 is not None else None,
                "raw": _jsonable(builtin_raw) if builtin_raw is not None else None,
                "error": builtin_error,
            },
        }
        iteration_payloads.append(iter_payload)

        if best_for_iter is not None:
            current_score = float(best_for_iter["metrics"]["exact_canonical"]["f1"])
            score_name = "exact_canonical_f1"
        elif builtin_f1 is not None:
            current_score = float(builtin_f1)
            score_name = "builtin_f1"
        else:
            raise RuntimeError("No evaluation metrics were produced. Check --eval-mode and --skip-builtin-eval.")

        if builtin_f1 is None and best_for_iter is not None:
            print(
                f"[ok] iteration={iter_num} best_threshold={best_for_iter['threshold']} {score_name}={current_score:.4f}"
            )
        elif builtin_f1 is not None and best_for_iter is not None:
            print(
                f"[ok] iteration={iter_num} best_threshold={best_for_iter['threshold']} "
                f"exact_canonical_f1={best_for_iter['metrics']['exact_canonical']['f1']:.4f} builtin_f1={builtin_f1:.4f}"
            )
        else:
            print(f"[ok] iteration={iter_num} {score_name}={current_score:.4f}")

        if current_score > best_exact_f1:
            best_exact_f1 = current_score
            best_iteration_payload = iter_payload

    assert best_iteration_payload is not None
    generated_at = datetime.now(tz=UTC)
    report = {
        "report_version": "1.0",
        "generated_at_utc": generated_at.isoformat(),
        "dataset": {
            "name": bundle.dataset_name,
            "splits": bundle.splits,
            "sample_count": len(bundle.samples),
            "record_count": len(bundle.training_records),
            "labels": bundle.labels,
            "stats": bundle.stats,
        },
        "config": {
            "base_model_initial": args.base_model,
            "device": args.device,
            "precision": args.precision,
            "iterations": args.iterations,
            "epoch_schedule": epoch_schedule,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "flat_ner": args.flat_ner,
            "thresholds": thresholds,
            "skip_overlap_metrics": args.skip_overlap_metrics,
            "skip_per_label_metrics": args.skip_per_label_metrics,
        },
        "best_iteration": best_iteration_payload,
        "iterations": iteration_payloads,
    }

    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_root / f"scanpatch_gliner_finetune_report_{stamp}.json"
    md_path = output_root / f"scanpatch_gliner_finetune_report_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    best_result = report["best_iteration"]["best_threshold_result"]
    best = best_result["metrics"] if best_result is not None else None
    best_builtin_f1 = report["best_iteration"]["builtin_eval"]["f1"]
    md_lines = [
        "# GLiNER Fine-tuning Pipeline Report",
        "",
        f"- Dataset: `{bundle.dataset_name}`",
        f"- Splits: `{', '.join(bundle.splits)}`",
        f"- Samples: `{len(bundle.samples)}`",
        f"- Records: `{len(bundle.training_records)}`",
        f"- Labels: `{', '.join(bundle.labels)}`",
        f"- Iterations: `{args.iterations}`",
        "",
        "## Best Iteration",
        "",
        f"- Iteration: `{report['best_iteration']['iteration']}`",
        f"- Model path: `{report['best_iteration']['finetune']['final_model_dir']}`",
        f"- Threshold: `{best_result['threshold'] if best_result is not None else None}`",
        f"- Exact canonical F1: `{best['exact_canonical']['f1'] if best is not None else None}`",
        f"- Overlap canonical F1: `{best['overlap_canonical']['f1'] if best is not None else None}`",
        f"- Built-in GLiNER eval F1: `{best_builtin_f1}`",
        "",
        "## Notes",
        "",
        "- This report is train-on-all/eval-on-the-same-data by design for MVP feasibility.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[ok] report json: {json_path}")
    print(f"[ok] report md: {md_path}")
    print(f"[ok] best model: {report['best_iteration']['finetune']['final_model_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
