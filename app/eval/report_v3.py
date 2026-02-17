from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from app.eval.aggregate import metric_payload
from app.eval.metrics import EvaluationAggregate
from app.eval.types import EvalSpan, MetricCounts


def _now_utc_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _render_metric_line(name: str, m: dict[str, Any]) -> str:
    return (
        f"- `{name}`: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, "
        f"Residual={m['residual_miss_ratio']:.4f}, TP={m['true_positives']}, FP={m['false_positives']}, FN={m['false_negatives']}"
    )


def aggregate_payload(aggregate: EvaluationAggregate) -> dict[str, Any]:
    return {
        "exact_agnostic": metric_payload(aggregate.exact_agnostic),
        "overlap_agnostic": metric_payload(aggregate.overlap_agnostic),
        "exact_canonical": metric_payload(aggregate.exact_canonical),
        "overlap_canonical": metric_payload(aggregate.overlap_canonical),
        "char_canonical": metric_payload(aggregate.char_canonical),
        "token_canonical": metric_payload(aggregate.token_canonical),
        "per_label_exact": {k: metric_payload(v) for k, v in sorted(aggregate.per_label_exact.items())},
        "per_label_char": {k: metric_payload(v) for k, v in sorted(aggregate.per_label_char.items())},
    }


def render_summary_md(report: dict[str, Any]) -> str:
    suite = report["suite"]
    env = report["environment"]
    summ = report["results"]["suite_summary"]
    lines: list[str] = [
        "# Guardrails Evaluation Report",
        "",
        f"- Suite: `{suite['suite_id']}`",
        f"- Split(s): `{', '.join(suite.get('splits', []))}`",
        f"- Datasets: `{len(report['results']['by_dataset'])}`",
        f"- Policy: `{suite['policy_name']}`",
        f"- Runtime: `{env['runtime_mode']}` (cpu_device={env.get('cpu_device','') or 'n/a'})",
        f"- Generated: `{report['generated_at_utc']}`",
        "",
        "## Headline KPI",
        "",
        f"- risk_weighted_macro_char_recall: `{summ['headline']['risk_weighted_macro_char_recall']}`",
        f"- risk_weighted_macro_char_residual_miss_ratio: `{summ['headline']['risk_weighted_macro_char_residual_miss_ratio']}`",
        "",
        "## Suite Micro Metrics (Canonical)",
        "",
    ]

    micro = summ["micro"]
    lines.append(_render_metric_line("overlap_canonical", micro["overlap_canonical"]))
    lines.append(_render_metric_line("char_canonical", micro["char_canonical"]))
    lines.append(_render_metric_line("exact_canonical", micro["exact_canonical"]))
    lines.append("")
    lines.append("## By Label (Char Canonical)")
    lines.append("")
    by_label = report["results"]["by_label"]
    for label, payload in sorted(by_label.items()):
        m = payload["char_canonical"]
        support = m.get("support_gold", 0)
        lines.append(
            f"- `{label}`: R={m['recall']:.4f} (gold={support}, weight={payload.get('weight', 1.0)}) "
            f"P={m['precision']:.4f} F1={m['f1']:.4f}"
        )

    lines.append("")
    lines.append("## By Dataset (Micro Canonical)")
    lines.append("")
    for item in report["results"]["by_dataset"]:
        ds_id = item["dataset_id"]
        ds_split = item["split"]
        samples = item["sample_count"]
        m = item["metrics"]["micro"]["overlap_canonical"]
        lines.append(f"### `{ds_id}` (`{ds_split}`)")
        lines.append("")
        lines.append(f"- Samples: `{samples}`")
        lines.append(f"- overlap_canonical: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
        m = item["metrics"]["micro"]["char_canonical"]
        lines.append(f"- char_canonical: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
        lines.append("")

    if report.get("comparison"):
        lines.append("## Comparison")
        lines.append("")
        comp = report["comparison"]
        lines.append(f"- Base report: `{comp['base_report_path']}`")
        for k, v in comp.get("headline_delta", {}).items():
            lines.append(f"- {k}: `{v}`")
        top_regressions = comp.get("top_regressions") or []
        top_improvements = comp.get("top_improvements") or []
        if top_regressions:
            lines.append("")
            lines.append("### Top Regressions (Label Char Recall)")
            lines.append("")
            for item in top_regressions:
                lines.append(f"- `{item['label']}`: {item['delta']:+.6f}")
        if top_improvements:
            lines.append("")
            lines.append("### Top Improvements (Label Char Recall)")
            lines.append("")
            for item in top_improvements:
                lines.append(f"- `{item['label']}`: {item['delta']:+.6f}")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
