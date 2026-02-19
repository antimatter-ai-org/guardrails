from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.eval.reporting.render_md import render_report_markdown


def write_report_files(*, report: dict[str, Any], output_dir: str, run_id: str) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve() / run_id
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_report_markdown(report), encoding="utf-8")

    metrics_csv_path = out / "metrics.csv"
    _write_metrics_csv(report=report, path=metrics_csv_path)

    return {
        "report_json": str(json_path),
        "report_md": str(md_path),
        "metrics_csv": str(metrics_csv_path),
        "run_dir": str(out),
    }


def _write_metrics_csv(*, report: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []

    tasks = report.get("tasks") or {}
    if isinstance(tasks, dict):
        span = tasks.get("span_detection")
        if isinstance(span, dict):
            combined = ((span.get("metrics") or {}).get("combined")) or {}
            for fam in ("exact_canonical", "overlap_canonical", "char_canonical", "token_canonical"):
                metric = combined.get(fam)
                if isinstance(metric, dict):
                    rows.append(
                        {
                            "task": "span_detection",
                            "scope": "combined",
                            "dataset": "",
                            "label": "",
                            "metric_family": fam,
                            **{k: metric.get(k) for k in ("precision", "recall", "f1", "true_positives", "false_positives", "false_negatives")},
                        }
                    )

            per_label_char = ((span.get("metrics") or {}).get("combined") or {}).get("per_label_char") or {}
            if isinstance(per_label_char, dict):
                for label, metric in per_label_char.items():
                    if not isinstance(metric, dict):
                        continue
                    rows.append(
                        {
                            "task": "span_detection",
                            "scope": "label",
                            "dataset": "",
                            "label": label,
                            "metric_family": "per_label_char",
                            **{k: metric.get(k) for k in ("precision", "recall", "f1", "true_positives", "false_positives", "false_negatives")},
                        }
                    )

        action = tasks.get("policy_action")
        if isinstance(action, dict):
            policies = action.get("policies") or {}
            if isinstance(policies, dict):
                for policy_name, payload in policies.items():
                    metrics = (payload or {}).get("metrics") or {}
                    if not isinstance(metrics, dict):
                        continue
                    rows.append(
                        {
                            "task": "policy_action",
                            "scope": "policy",
                            "dataset": "",
                            "label": policy_name,
                            "metric_family": "binary",
                            **{k: metrics.get(k) for k in ("precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "tp", "fp", "tn", "fn")},
                        }
                    )

        leakage = tasks.get("mask_leakage")
        if isinstance(leakage, dict):
            rows.append(
                {
                    "task": "mask_leakage",
                    "scope": "combined",
                    "dataset": "",
                    "label": "",
                    "metric_family": "leakage_fraction",
                    "value": leakage.get("leakage_fraction"),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        fieldnames = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

