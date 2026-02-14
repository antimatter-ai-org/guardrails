from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.eval.metrics import EvaluationAggregate
from app.eval.types import MetricCounts


def _metric_dict(metric: MetricCounts) -> dict[str, Any]:
    return {
        "true_positives": metric.true_positives,
        "false_positives": metric.false_positives,
        "false_negatives": metric.false_negatives,
        "precision": round(metric.precision, 6),
        "recall": round(metric.recall, 6),
        "f1": round(metric.f1, 6),
    }


def build_report_payload(
    *,
    dataset_name: str,
    split: str,
    sample_count: int,
    policy_name: str,
    policy_path: str,
    runtime_mode: str,
    elapsed_seconds: float,
    aggregate: EvaluationAggregate,
    errors_preview: list[dict[str, Any]],
) -> dict[str, Any]:
    samples_per_second = 0.0
    if elapsed_seconds > 0:
        samples_per_second = sample_count / elapsed_seconds

    return {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "dataset": {
            "name": dataset_name,
            "split": split,
            "sample_count": sample_count,
        },
        "evaluation": {
            "policy_name": policy_name,
            "policy_path": policy_path,
            "runtime_mode": runtime_mode,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "samples_per_second": round(samples_per_second, 6),
        },
        "metrics": {
            "exact_agnostic": _metric_dict(aggregate.exact_agnostic),
            "overlap_agnostic": _metric_dict(aggregate.overlap_agnostic),
            "exact_canonical": _metric_dict(aggregate.exact_canonical),
            "overlap_canonical": _metric_dict(aggregate.overlap_canonical),
            "per_label_exact": {
                label: _metric_dict(metric)
                for label, metric in sorted(aggregate.per_label_exact.items())
            },
        },
        "errors_preview": errors_preview,
    }


def render_markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        f"# Guardrails Evaluation Report",
        "",
        f"- Dataset: `{report['dataset']['name']}` (`{report['dataset']['split']}`)",
        f"- Samples: `{report['dataset']['sample_count']}`",
        f"- Policy: `{report['evaluation']['policy_name']}`",
        f"- Runtime: `{report['evaluation']['runtime_mode']}`",
        f"- Eval Mode: `{report['evaluation'].get('mode', 'baseline')}`",
        f"- Elapsed: `{report['evaluation']['elapsed_seconds']}s`",
        f"- Throughput: `{report['evaluation']['samples_per_second']}` samples/s",
        "",
        "## Aggregate Metrics",
        "",
    ]

    for metric_name in ("exact_agnostic", "overlap_agnostic", "exact_canonical", "overlap_canonical"):
        metric = report["metrics"][metric_name]
        lines.append(
            (
                f"- `{metric_name}`: "
                f"P={metric['precision']:.4f}, "
                f"R={metric['recall']:.4f}, "
                f"F1={metric['f1']:.4f}, "
                f"TP={metric['true_positives']}, FP={metric['false_positives']}, FN={metric['false_negatives']}"
            )
        )

    lines.append("")
    lines.append("## Per-Label (Exact Canonical)")
    lines.append("")
    for label, metric in report["metrics"]["per_label_exact"].items():
        lines.append(
            (
                f"- `{label}`: P={metric['precision']:.4f}, "
                f"R={metric['recall']:.4f}, F1={metric['f1']:.4f}, "
                f"TP={metric['true_positives']}, FP={metric['false_positives']}, FN={metric['false_negatives']}"
            )
        )

    cascade = report.get("evaluation", {}).get("cascade")
    if isinstance(cascade, dict):
        lines.append("")
        lines.append("## Cascade")
        lines.append("")
        lines.append(
            f"- Threshold: `{cascade.get('threshold')}`"
        )
        lines.append(
            f"- Escalated: `{cascade.get('escalated_samples')}` / `{report['dataset']['sample_count']}`"
        )

    return "\n".join(lines) + "\n"


def write_report_files(report: dict[str, Any], output_dir: str, report_slug: str) -> tuple[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    json_path = output / f"{report_slug}.json"
    md_path = output / f"{report_slug}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_summary(report), encoding="utf-8")
    return str(json_path), str(md_path)
