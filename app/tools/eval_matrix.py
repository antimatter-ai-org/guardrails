from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation matrix across multiple policy names.")
    parser.add_argument("--policy-path", default="configs/policy.yaml")
    parser.add_argument("--policy-name", action="append", required=True, help="Policy name (repeatable).")
    parser.add_argument("--dataset", action="append", default=None, help="Dataset name (repeatable).")
    parser.add_argument("--split", default="test")
    parser.add_argument("--env-file", default=".env.eval")
    parser.add_argument("--output-dir", default="reports/evaluations")
    parser.add_argument("--cache-dir", default=".eval_cache/hf")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--strict-split", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--warmup-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--comparison-output", default=None, help="Optional markdown path for base-vs-candidates comparison.")
    return parser.parse_args()


def _extract_json_report_path(stdout: str) -> str:
    for line in stdout.splitlines():
        prefix = "[ok] JSON report: "
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise RuntimeError("Could not find JSON report path in eval output.")


def _build_eval_command(args: argparse.Namespace, policy_name: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "app.eval.run",
        "--policy-path",
        args.policy_path,
        "--policy-name",
        policy_name,
        "--split",
        args.split,
        "--env-file",
        args.env_file,
        "--output-dir",
        args.output_dir,
        "--cache-dir",
        args.cache_dir,
        "--warmup-timeout-seconds",
        str(args.warmup_timeout_seconds),
    ]
    if args.dataset:
        for item in args.dataset:
            cmd.extend(["--dataset", item])
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.strict_split:
        cmd.append("--strict-split")
    if args.warmup_strict:
        cmd.append("--warmup-strict")
    return cmd


def main() -> int:
    args = _parse_args()
    policies = [item.strip() for item in args.policy_name if item and item.strip()]
    if not policies:
        raise ValueError("At least one --policy-name is required.")

    report_paths: list[str] = []
    for policy_name in policies:
        cmd = _build_eval_command(args, policy_name)
        print(f"[run] policy={policy_name} command={' '.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", flush=True)
        report_paths.append(_extract_json_report_path(completed.stdout))

    if len(report_paths) > 1:
        base = report_paths[0]
        compare_cmd = [
            sys.executable,
            "-m",
            "app.tools.compare_eval_reports",
            "--base",
            base,
        ]
        for candidate in report_paths[1:]:
            compare_cmd.extend(["--candidate", candidate])
        if args.comparison_output:
            compare_cmd.extend(["--output", args.comparison_output])
        print(f"[run] compare command={' '.join(compare_cmd)}", flush=True)
        completed = subprocess.run(compare_cmd, check=True, capture_output=True, text=True)
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", flush=True)

    print(f"[ok] matrix complete policies={', '.join(policies)}", flush=True)
    for report_path in report_paths:
        print(f"[ok] report: {report_path}", flush=True)
    if args.comparison_output:
        print(f"[ok] comparison: {Path(args.comparison_output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
