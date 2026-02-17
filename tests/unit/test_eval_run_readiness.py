from __future__ import annotations

from argparse import Namespace
from typing import Any

import pytest

from app.eval import run
from app.eval.suite_loader import DatasetSpec, SuiteSpec
from app import settings as settings_mod
import app.config as config_mod
import app.core.analysis.service as analysis_service_mod
import app.model_assets as model_assets_mod


class _FakePolicy:
    def __init__(self, *, analyzer_profile: str = "external_rich", min_score: float = 0.0) -> None:
        self.analyzer_profile = analyzer_profile
        self.min_score = min_score


class _FakeConfig:
    def __init__(self, policy: _FakePolicy) -> None:
        self.default_policy = "external_default"
        self.policies = {"external_default": policy}


class _WarmupService:
    def __init__(self, *, readiness_errors: dict[str, str] | None = None) -> None:
        self.readiness_errors = readiness_errors or {}
        self.ready_called = False

    def ensure_profile_runtimes_ready(self, *, profile_names: list[str], timeout_s: float) -> dict[str, str]:
        self.ready_called = True
        return self.readiness_errors

    def analyze_text(self, *, text: str, profile_name: str, policy_min_score: float) -> list[Any]:
        return []


def _args(tmp_path) -> Namespace:
    return Namespace(
        suite="guardrails-ru",
        dataset=None,
        tag=None,
        split=["fast"],
        full=False,
        policy_path="configs/policy.yaml",
        policy_name=None,
        cache_dir=str(tmp_path / ".eval_cache"),
        output_dir=str(tmp_path / "out"),
        env_file=str(tmp_path / ".env.eval"),
        hf_token_env="HF_TOKEN",
        refresh_collection=False,
        runtime_mode=None,
        cpu_device=None,
        weights_path="configs/eval/weights.yaml",
        gates_path="configs/eval/gates.yaml",
        enforce_gates=False,
        view=None,
        where=None,
        max_samples=1,
        seed=42,
        stratify_by=None,
        compare=None,
        errors_preview_limit=5,
        progress_every_samples=1000,
        progress_every_seconds=15.0,
    )


def test_eval_main_runs_runtime_readiness_before_loading_datasets(monkeypatch, tmp_path) -> None:
    service = _WarmupService()
    suite = SuiteSpec(
        suite_id="guardrails-ru",
        default_collection="",
        default_split="fast",
        scored_labels=("person",),
        datasets=(
            DatasetSpec(
                dataset_id="org/ds",
                format="privacy_mask_parquet_v1",
                text_field="source_text",
                mask_field="privacy_mask",
                annotated_labels=("person",),
                gold_label_mapping={"person": "person"},
                slice_fields=tuple(),
                tags=tuple(),
                notes="",
            ),
        ),
    )

    monkeypatch.setattr(run, "_parse_args", lambda: _args(tmp_path))
    monkeypatch.setattr(run, "load_env_file", lambda _path: None)
    monkeypatch.setattr(run, "load_suite", lambda _p: suite)
    monkeypatch.setattr(run, "load_weights", lambda _p: {"person": 1.0})
    monkeypatch.setattr(run, "load_gates", lambda _p: {})
    monkeypatch.setattr(config_mod, "load_policy_config", lambda _p: _FakeConfig(_FakePolicy()))
    monkeypatch.setattr(model_assets_mod, "apply_model_env", lambda **_kwargs: None)
    monkeypatch.setattr(analysis_service_mod, "PresidioAnalysisService", lambda _cfg: service)
    monkeypatch.setattr(settings_mod.settings, "pytriton_init_timeout_s", 0.01)

    def _load_hf_dataset(*, dataset_id, split, cache_paths, hf_token):
        assert service.ready_called is True
        return run.LoadedDataset(
            dataset_id=dataset_id,
            split=split,
            available_splits=("fast", "full"),
            fingerprint="fp",
            rows=[{"source_text": "hello", "privacy_mask": [{"start": 0, "end": 5, "label": "person"}]}],
        )

    monkeypatch.setattr(run, "load_hf_dataset", _load_hf_dataset)

    rc = run.main()
    assert rc == 0


def test_eval_main_fails_fast_when_runtime_readiness_fails(monkeypatch, tmp_path) -> None:
    service = _WarmupService(readiness_errors={"external_rich:gliner": "runtime is not ready"})

    monkeypatch.setattr(run, "_parse_args", lambda: _args(tmp_path))
    monkeypatch.setattr(run, "load_env_file", lambda _path: None)
    monkeypatch.setattr(
        run,
        "load_suite",
        lambda _p: SuiteSpec(
            suite_id="guardrails-ru",
            default_collection="",
            default_split="fast",
            scored_labels=("person",),
            datasets=(
                DatasetSpec(
                    dataset_id="org/ds",
                    format="privacy_mask_parquet_v1",
                    text_field="source_text",
                    mask_field="privacy_mask",
                    annotated_labels=("person",),
                    gold_label_mapping={"person": "person"},
                    slice_fields=tuple(),
                    tags=tuple(),
                    notes="",
                ),
            ),
        ),
    )
    monkeypatch.setattr(run, "load_weights", lambda _p: {})
    monkeypatch.setattr(run, "load_gates", lambda _p: {})
    monkeypatch.setattr(config_mod, "load_policy_config", lambda _p: _FakeConfig(_FakePolicy()))
    monkeypatch.setattr(model_assets_mod, "apply_model_env", lambda **_kwargs: None)
    monkeypatch.setattr(analysis_service_mod, "PresidioAnalysisService", lambda _cfg: service)
    monkeypatch.setattr(settings_mod.settings, "pytriton_init_timeout_s", 0.01)
    monkeypatch.setattr(run, "load_hf_dataset", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not load datasets")))

    with pytest.raises(RuntimeError, match="model runtime readiness check failed"):
        run.main()
