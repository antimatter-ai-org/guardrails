from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from app.eval import run
from app.eval.types import EvalSample


class _FakePolicy:
    def __init__(self, *, analyzer_profile: str = "external_rich", min_score: float = 0.0) -> None:
        self.analyzer_profile = analyzer_profile
        self.min_score = min_score


class _FakeConfig:
    def __init__(self, policy: _FakePolicy) -> None:
        self.default_policy = "external_default"
        self.policies = {"external_default": policy}


class _FakeAdapter:
    supports_synthetic_split = False

    def __init__(self, samples: list[EvalSample]) -> None:
        self._samples = samples

    def load_samples(
        self,
        *,
        split: str,
        cache_dir: str,
        hf_token: str | None,
        synthetic_test_size: float,
        synthetic_split_seed: int,
        max_samples: int | None,
    ) -> list[EvalSample]:
        if max_samples is None:
            return self._samples
        return self._samples[:max_samples]


class _WarmupService:
    def __init__(self, *, warmup_errors: dict[str, str] | None = None) -> None:
        self.warmup_errors = warmup_errors or {}
        self.warmup_calls: list[tuple[list[str], float]] = []

    def warm_up_profile_runtimes(self, *, profile_names: list[str], timeout_s: float) -> dict[str, str]:
        self.warmup_calls.append((profile_names, timeout_s))
        return self.warmup_errors

    def analyze_text(self, *, text: str, profile_name: str, policy_min_score: float) -> list[Any]:
        return []


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        dataset=["dummy/dataset"],
        split="test",
        policy_path="configs/policy.yaml",
        policy_name=None,
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "out"),
        env_file=str(tmp_path / ".env.eval"),
        hf_token_env="HF_TOKEN",
        strict_split=False,
        synthetic_test_size=0.2,
        synthetic_split_seed=42,
        max_samples=1,
        errors_preview_limit=5,
        progress_every_samples=1000,
        progress_every_seconds=15.0,
    )


def test_eval_main_warms_runtime_models_before_processing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _WarmupService()
    sample = EvalSample(sample_id="1", text="hello", gold_spans=[])

    monkeypatch.setattr(run, "_parse_args", lambda: _args(tmp_path))
    monkeypatch.setattr(run, "load_env_file", lambda _path: None)
    monkeypatch.setattr(run, "_configure_hf_cache", lambda _cache_dir: None)
    monkeypatch.setattr(run, "apply_model_env", lambda **_kwargs: None)
    monkeypatch.setattr(run, "load_policy_config", lambda _path: _FakeConfig(_FakePolicy()))
    monkeypatch.setattr(run, "PresidioAnalysisService", lambda _cfg: service)
    monkeypatch.setattr(run, "_resolve_dataset_split", lambda **_kwargs: ("test", ["test"]))
    monkeypatch.setattr(run, "get_dataset_adapter", lambda _name: _FakeAdapter([sample]))
    monkeypatch.setattr(run.settings, "pytriton_init_timeout_s", 33.0)
    monkeypatch.setattr(run, "write_report_files", lambda *_args, **_kwargs: ("report.json", "report.md"))

    rc = run.main()

    assert rc == 0
    assert service.warmup_calls == [(["external_rich"], 33.0)]


def test_eval_main_fails_fast_when_runtime_warmup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _WarmupService(warmup_errors={"external_rich:gliner": "runtime is not ready"})

    monkeypatch.setattr(run, "_parse_args", lambda: _args(tmp_path))
    monkeypatch.setattr(run, "load_env_file", lambda _path: None)
    monkeypatch.setattr(run, "_configure_hf_cache", lambda _cache_dir: None)
    monkeypatch.setattr(run, "apply_model_env", lambda **_kwargs: None)
    monkeypatch.setattr(run, "load_policy_config", lambda _path: _FakeConfig(_FakePolicy()))
    monkeypatch.setattr(run, "PresidioAnalysisService", lambda _cfg: service)
    monkeypatch.setattr(run.settings, "pytriton_init_timeout_s", 17.0)
    monkeypatch.setattr(run, "get_dataset_adapter", lambda _name: (_ for _ in ()).throw(AssertionError("should not load datasets")))

    with pytest.raises(RuntimeError, match="model runtime warm-up failed"):
        run.main()

    assert service.warmup_calls == [(["external_rich"], 17.0)]
