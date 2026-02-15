from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _load_report(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_commit_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _report_entry(path: str, report: dict[str, Any]) -> dict[str, Any]:
    dataset = report.get("dataset", {}) if isinstance(report.get("dataset"), dict) else {}
    evaluation = report.get("evaluation", {}) if isinstance(report.get("evaluation"), dict) else {}
    datasets = report.get("datasets", [])
    if not isinstance(datasets, list):
        datasets = []
    return {
        "path": str(Path(path)),
        "generated_at_utc": report.get("generated_at_utc"),
        "dataset": {
            "name": dataset.get("name"),
            "split": dataset.get("split"),
            "sample_count": dataset.get("sample_count"),
        },
        "evaluation": {
            "policy_name": evaluation.get("policy_name"),
            "policy_path": evaluation.get("policy_path"),
            "runtime_mode": evaluation.get("runtime_mode"),
            "mode": evaluation.get("mode", "baseline"),
        },
        "datasets_count": len(datasets),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create baseline manifest for a set of eval reports.")
    parser.add_argument("--report", action="append", default=[], help="Report JSON path (repeatable).")
    parser.add_argument(
        "--output",
        default="reports/evaluations/baseline_manifest.json",
        help="Output manifest path.",
    )
    parser.add_argument("--notes", default="", help="Optional free-form notes.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.report:
        raise ValueError("At least one --report path is required.")

    report_paths = [str(Path(path)) for path in args.report]
    report_entries: list[dict[str, Any]] = []
    for path in report_paths:
        report_entries.append(_report_entry(path, _load_report(path)))

    manifest = {
        "manifest_version": "1.0",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_commit_sha": _git_commit_sha(),
        "reports": report_entries,
        "notes": args.notes,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] baseline manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
