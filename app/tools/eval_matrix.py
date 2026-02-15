from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


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
    parser.add_argument(
        "--ablate-recognizer",
        action="append",
        default=[],
        help="Recognizer id to ablate from each selected policy (repeatable).",
    )
    parser.add_argument(
        "--ablation-policy-dir",
        default="reports/evaluations/_ablation_policies",
        help="Directory for generated ablation policy files.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
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
    if args.resume:
        cmd.append("--resume")
    return cmd


def _slug(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum() or char in {"_", "-"}:
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def _build_ablation_policy(
    *,
    source_policy_path: str,
    source_policy_name: str,
    recognizer_id: str,
    output_dir: str,
) -> tuple[str, str]:
    src_path = Path(source_policy_path)
    raw = yaml.safe_load(src_path.read_text(encoding="utf-8")) or {}
    policies = raw.get("policies", {})
    profiles = raw.get("analyzer_profiles", {})
    if source_policy_name not in policies:
        raise KeyError(f"Policy '{source_policy_name}' not found in {source_policy_path}")

    source_policy = policies[source_policy_name]
    profile_name = str(source_policy.get("analyzer_profile", ""))
    if profile_name not in profiles:
        raise KeyError(f"Analyzer profile '{profile_name}' not found in {source_policy_path}")

    source_profile = profiles[profile_name]
    recognizers = list(source_profile.get("analysis", {}).get("recognizers", []) or [])
    ablated = [item for item in recognizers if str(item) != recognizer_id]

    profile_slug = _slug(profile_name)
    recognizer_slug = _slug(recognizer_id)
    new_profile_name = f"{profile_slug}__ablate__{recognizer_slug}"
    new_policy_name = f"{_slug(source_policy_name)}__ablate__{recognizer_slug}"

    new_profile = copy.deepcopy(source_profile)
    new_profile.setdefault("analysis", {})
    new_profile["analysis"]["recognizers"] = ablated
    profiles[new_profile_name] = new_profile

    new_policy = copy.deepcopy(source_policy)
    new_policy["analyzer_profile"] = new_profile_name
    policies[new_policy_name] = new_policy

    raw["policies"] = policies
    raw["analyzer_profiles"] = profiles

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src_path.stem}__{_slug(source_policy_name)}__ablate__{recognizer_slug}.yaml"
    out_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(out_path), new_policy_name


def _run_eval_command(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", flush=True)
    return _extract_json_report_path(completed.stdout)


def main() -> int:
    args = _parse_args()
    policies = [item.strip() for item in args.policy_name if item and item.strip()]
    if not policies:
        raise ValueError("At least one --policy-name is required.")
    ablations = [item.strip() for item in args.ablate_recognizer if item and item.strip()]

    run_results: list[tuple[str, str]] = []
    for policy_name in policies:
        cmd = _build_eval_command(args, policy_name)
        print(f"[run] policy={policy_name} command={' '.join(cmd)}", flush=True)
        report_path = _run_eval_command(cmd)
        run_results.append((policy_name, report_path))

        for recognizer_id in ablations:
            ablation_policy_path, ablation_policy_name = _build_ablation_policy(
                source_policy_path=args.policy_path,
                source_policy_name=policy_name,
                recognizer_id=recognizer_id,
                output_dir=args.ablation_policy_dir,
            )
            ablation_cmd = _build_eval_command(args, ablation_policy_name)
            ablation_cmd[ablation_cmd.index("--policy-path") + 1] = ablation_policy_path
            print(
                f"[run] policy={policy_name} ablation={recognizer_id} command={' '.join(ablation_cmd)}",
                flush=True,
            )
            report_path = _run_eval_command(ablation_cmd)
            run_results.append((f"{policy_name}__ablate__{recognizer_id}", report_path))

    report_paths = [path for _, path in run_results]
    if len(run_results) > 1:
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
    for label, report_path in run_results:
        print(f"[ok] report[{label}]: {report_path}", flush=True)
    if args.comparison_output:
        print(f"[ok] comparison: {Path(args.comparison_output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
