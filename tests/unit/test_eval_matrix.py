from __future__ import annotations

import types

from app.tools import eval_matrix


def test_build_eval_command_includes_selected_flags() -> None:
    args = types.SimpleNamespace(
        policy_path="configs/policy.yaml",
        split="test",
        env_file=".env.eval",
        output_dir="reports/evaluations",
        cache_dir=".eval_cache/hf",
        warmup_timeout_seconds=42.0,
        dataset=["a/b"],
        max_samples=100,
        strict_split=True,
        warmup_strict=True,
    )

    cmd = eval_matrix._build_eval_command(args, "external_default")

    assert "--policy-name" in cmd
    assert "external_default" in cmd
    assert "--dataset" in cmd
    assert "a/b" in cmd
    assert "--max-samples" in cmd
    assert "100" in cmd
    assert "--strict-split" in cmd
    assert "--warmup-strict" in cmd


def test_extract_json_report_path_reads_ok_line() -> None:
    stdout = "\n".join(
        [
            "[ok] Datasets: scanpatch",
            "[ok] JSON report: reports/evaluations/eval_x.json",
        ]
    )
    assert eval_matrix._extract_json_report_path(stdout) == "reports/evaluations/eval_x.json"
