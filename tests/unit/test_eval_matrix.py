from __future__ import annotations

import types
from pathlib import Path

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
        resume=True,
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
    assert "--resume" in cmd


def test_extract_json_report_path_reads_ok_line() -> None:
    stdout = "\n".join(
        [
            "[ok] Datasets: scanpatch",
            "[ok] JSON report: reports/evaluations/eval_x.json",
        ]
    )
    assert eval_matrix._extract_json_report_path(stdout) == "reports/evaluations/eval_x.json"


def test_build_ablation_policy_creates_variant_with_removed_recognizer(tmp_path: Path) -> None:
    src = tmp_path / "policy.yaml"
    src.write_text(
        """
default_policy: external_default
policies:
  external_default:
    mode: mask
    analyzer_profile: p1
    min_score: 0.5
analyzer_profiles:
  p1:
    language:
      default: ru
      supported: [ru, en]
      detection: auto
    analysis:
      backend: presidio
      nlp_engine: none
      use_builtin_recognizers: false
      recognizers: [a, b, c]
recognizer_definitions:
  a: {type: regex, enabled: true, params: {}}
  b: {type: regex, enabled: true, params: {}}
  c: {type: regex, enabled: true, params: {}}
""".strip(),
        encoding="utf-8",
    )

    out_path, policy_name = eval_matrix._build_ablation_policy(
        source_policy_path=str(src),
        source_policy_name="external_default",
        recognizer_id="b",
        output_dir=str(tmp_path / "out"),
    )

    assert Path(out_path).exists()
    assert policy_name.endswith("__ablate__b")
    text = Path(out_path).read_text(encoding="utf-8")
    assert "recognizers:" in text
    assert "- a" in text
    assert "- c" in text
