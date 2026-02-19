from __future__ import annotations

from pathlib import Path

import pytest

from app.eval import cli


def test_eval_report_includes_nemotron_setting_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "load_env_file", lambda _path: None)
    monkeypatch.setattr(
        cli,
        "load_eval_registry",
        lambda _path: type(
            "_R",
            (),
            {"suites": {"guardrails_ru": type("_S", (), {"name": "guardrails_ru", "datasets": [], "default_split": "fast"})()}, "datasets": {}},
        )(),
    )
    monkeypatch.setattr(cli, "_resolve_suite_and_datasets", lambda *_args, **_kwargs: ("guardrails_ru", [], "fast"))
    fake_policy = type(
        "_P",
        (),
        {
            "mode": "mask",
            "analyzer_profile": "external",
            "min_score": 0.0,
        },
    )()
    monkeypatch.setattr(
        cli,
        "load_policy_config",
        lambda _path: type("_Cfg", (), {"default_policy": "external", "policies": {"external": fake_policy}})(),
    )
    monkeypatch.setattr(cli, "PresidioAnalysisService", lambda _cfg: object())
    monkeypatch.setattr(cli, "_ensure_runtime_ready", lambda **_kwargs: None)

    monkeypatch.setattr(cli.settings, "runtime_mode", "cpu")
    monkeypatch.setattr(cli.settings, "enable_nemotron", True)

    monkeypatch.setenv("GR_ENABLE_NEMOTRON", "true")
    monkeypatch.setenv("GR_RUNTIME_MODE", "cpu")

    args = type(
        "_Args",
        (),
        {
            "registry_path": str(tmp_path / "registry.yaml"),
            "suite": "guardrails_ru",
            "dataset": None,
            "split": "fast",
            "tasks": "policy_action",
            "policy_path": "configs/policy.yaml",
            "policy_name": None,
            "action_policies": "external",
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "out"),
            "env_file": str(tmp_path / ".env.eval"),
            "hf_token_env": "HF_TOKEN",
            "offline": False,
            "runtime": "cpu",
            "cpu_device": "auto",
            "subset": "all",
            "sampler": "none",
            "sampler_size": None,
            "sampler_seed": 42,
            "min_per_label": None,
            "max_samples": None,
            "errors_preview_limit": 1,
            "progress_every_samples": 1000,
            "progress_every_seconds": 1.0,
            "workers": 1,
        },
    )()
    monkeypatch.setattr(cli, "_parse_args", lambda: args)

    outputs = {}

    def capture_report(*, report, output_dir, run_id):  # noqa: ANN001
        outputs["report"] = report
        return {"report_json": str(tmp_path / "report.json")}

    monkeypatch.setattr(cli, "write_report_files", capture_report)

    assert cli.main() == 0
    report = outputs["report"]
    assert report["run"]["settings"]["enable_nemotron"] is True
    assert report["run"]["env"]["GR_ENABLE_NEMOTRON"] == "true"
