from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_report(path: str) -> dict[str, Any]:
    report_path = Path(path)
    return json.loads(report_path.read_text(encoding="utf-8"))


def _metric_f1(report: dict[str, Any], metric_name: str) -> float:
    metric = report.get("metrics", {}).get(metric_name, {})
    return float(metric.get("f1", 0.0))


def _extract_dataset_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = report.get("datasets")
    if isinstance(datasets, list) and datasets:
        return [item for item in datasets if isinstance(item, dict)]

    dataset = report.get("dataset", {})
    if not isinstance(dataset, dict):
        dataset = {}
    return [
        {
            "name": str(dataset.get("name", "unknown")),
            "split": str(dataset.get("split", "unknown")),
            "sample_count": int(dataset.get("sample_count", 0) or 0),
            "metrics": report.get("metrics", {}),
        }
    ]


def _extract_dataset_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _extract_dataset_rows(report):
        name = str(row.get("name", "unknown"))
        split = str(row.get("split", "unknown"))
        key = f"{name}::{split}"
        result[key] = row
    return result


def _extract_per_label(report: dict[str, Any]) -> dict[str, float]:
    payload = report.get("metrics", {}).get("per_label_exact", {})
    if not isinstance(payload, dict):
        return {}
    values: dict[str, float] = {}
    for label, metric in payload.items():
        if not isinstance(metric, dict):
            continue
        values[str(label)] = float(metric.get("f1", 0.0))
    return values


def _delta(candidate: float, base: float) -> float:
    return candidate - base


def _format_delta(value: float) -> str:
    return f"{value:+.4f}"


def _render_candidate_vs_base(
    *,
    base_report: dict[str, Any],
    base_path: str,
    candidate_report: dict[str, Any],
    candidate_path: str,
) -> str:
    lines: list[str] = []
    lines.append(f"## Candidate: `{candidate_path}`")
    lines.append("")
    lines.append(f"- Base: `{base_path}`")
    lines.append("")
    lines.append("| Scope | Metric | Base F1 | Candidate F1 | Delta |")
    lines.append("|---|---:|---:|---:|---:|")

    for metric_name in ("exact_canonical", "overlap_canonical"):
        base_f1 = _metric_f1(base_report, metric_name)
        candidate_f1 = _metric_f1(candidate_report, metric_name)
        lines.append(
            f"| combined | {metric_name} | {base_f1:.4f} | {candidate_f1:.4f} | {_format_delta(_delta(candidate_f1, base_f1))} |"
        )

    base_datasets = _extract_dataset_map(base_report)
    candidate_datasets = _extract_dataset_map(candidate_report)
    all_dataset_keys = sorted(set(base_datasets) | set(candidate_datasets))
    if all_dataset_keys:
        lines.append("")
        lines.append("| Dataset | Metric | Base F1 | Candidate F1 | Delta |")
        lines.append("|---|---:|---:|---:|---:|")
        for key in all_dataset_keys:
            base_row = base_datasets.get(key, {})
            candidate_row = candidate_datasets.get(key, {})
            for metric_name in ("exact_canonical", "overlap_canonical"):
                base_f1 = float(base_row.get("metrics", {}).get(metric_name, {}).get("f1", 0.0))
                candidate_f1 = float(candidate_row.get("metrics", {}).get(metric_name, {}).get("f1", 0.0))
                lines.append(
                    f"| {key} | {metric_name} | {base_f1:.4f} | {candidate_f1:.4f} | {_format_delta(_delta(candidate_f1, base_f1))} |"
                )

    base_labels = _extract_per_label(base_report)
    candidate_labels = _extract_per_label(candidate_report)
    all_labels = sorted(set(base_labels) | set(candidate_labels))
    if all_labels:
        lines.append("")
        lines.append("| Label | Base F1 | Candidate F1 | Delta |")
        lines.append("|---|---:|---:|---:|")
        for label in all_labels:
            base_f1 = float(base_labels.get(label, 0.0))
            candidate_f1 = float(candidate_labels.get(label, 0.0))
            lines.append(
                f"| {label} | {base_f1:.4f} | {candidate_f1:.4f} | {_format_delta(_delta(candidate_f1, base_f1))} |"
            )

    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evaluation reports and render markdown deltas.")
    parser.add_argument("--base", required=True, help="Path to baseline report JSON.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Path to candidate report JSON (repeatable).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write markdown output. If omitted, output is printed.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.candidate:
        raise ValueError("At least one --candidate report path is required.")

    base_path = str(Path(args.base))
    base_report = _load_report(base_path)
    candidate_paths = [str(Path(path)) for path in args.candidate]
    blocks = [
        "# Evaluation Comparison",
        "",
        f"- Base report: `{base_path}`",
        f"- Candidate reports: `{', '.join(candidate_paths)}`",
        "",
    ]

    for candidate_path in candidate_paths:
        candidate_report = _load_report(candidate_path)
        blocks.append(
            _render_candidate_vs_base(
                base_report=base_report,
                base_path=base_path,
                candidate_report=candidate_report,
                candidate_path=candidate_path,
            )
        )

    markdown = "\n".join(blocks).rstrip() + "\n"

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"[ok] comparison markdown: {output_path}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
