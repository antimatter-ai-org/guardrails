from __future__ import annotations

import json
from pathlib import Path

from app.tools import compare_eval_reports, create_eval_manifest


def _report(
    *,
    dataset_name: str,
    split: str,
    exact_f1: float,
    overlap_f1: float,
    per_label: dict[str, float] | None = None,
) -> dict:
    labels = per_label or {}
    return {
        "report_version": "2.0",
        "generated_at_utc": "2026-02-15T12:00:00+00:00",
        "dataset": {
            "name": dataset_name,
            "split": split,
            "sample_count": 10,
        },
        "evaluation": {
            "policy_name": "external_default",
            "policy_path": "configs/policy.yaml",
            "runtime_mode": "cpu",
        },
        "metrics": {
            "exact_canonical": {"f1": exact_f1},
            "overlap_canonical": {"f1": overlap_f1},
            "per_label_exact": {
                key: {"f1": value}
                for key, value in labels.items()
            },
        },
    }


def test_compare_renders_combined_dataset_and_label_deltas() -> None:
    base = _report(
        dataset_name="demo/a",
        split="test",
        exact_f1=0.2,
        overlap_f1=0.5,
        per_label={"location": 0.1, "person": 0.3},
    )
    candidate = _report(
        dataset_name="demo/a",
        split="test",
        exact_f1=0.25,
        overlap_f1=0.45,
        per_label={"location": 0.2, "person": 0.2},
    )

    rendered = compare_eval_reports._render_candidate_vs_base(
        base_report=base,
        base_path="base.json",
        candidate_report=candidate,
        candidate_path="candidate.json",
    )

    assert "| combined | exact_canonical | 0.2000 | 0.2500 | +0.0500 |" in rendered
    assert "| demo/a::test | overlap_canonical | 0.5000 | 0.4500 | -0.0500 |" in rendered
    assert "| location | 0.1000 | 0.2000 | +0.1000 |" in rendered


def test_compare_supports_multi_dataset_payload() -> None:
    base = _report(dataset_name="combined", split="test", exact_f1=0.2, overlap_f1=0.2)
    base["datasets"] = [
        {"name": "a", "split": "test", "sample_count": 3, "metrics": {"exact_canonical": {"f1": 0.1}, "overlap_canonical": {"f1": 0.2}}},
        {"name": "b", "split": "test", "sample_count": 4, "metrics": {"exact_canonical": {"f1": 0.3}, "overlap_canonical": {"f1": 0.4}}},
    ]
    candidate = _report(dataset_name="combined", split="test", exact_f1=0.5, overlap_f1=0.6)
    candidate["datasets"] = [
        {"name": "a", "split": "test", "sample_count": 3, "metrics": {"exact_canonical": {"f1": 0.4}, "overlap_canonical": {"f1": 0.5}}},
        {"name": "b", "split": "test", "sample_count": 4, "metrics": {"exact_canonical": {"f1": 0.6}, "overlap_canonical": {"f1": 0.7}}},
    ]

    rendered = compare_eval_reports._render_candidate_vs_base(
        base_report=base,
        base_path="base.json",
        candidate_report=candidate,
        candidate_path="candidate.json",
    )
    assert "| a::test | exact_canonical | 0.1000 | 0.4000 | +0.3000 |" in rendered
    assert "| b::test | overlap_canonical | 0.4000 | 0.7000 | +0.3000 |" in rendered


def test_manifest_includes_report_metadata(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "r.json"
    report_payload = _report(dataset_name="scanpatch/pii", split="test", exact_f1=0.3, overlap_f1=0.6)
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    monkeypatch.setattr(create_eval_manifest, "_git_commit_sha", lambda: "abc123")
    entry = create_eval_manifest._report_entry(str(report_path), report_payload)

    assert entry["path"] == str(report_path)
    assert entry["dataset"]["name"] == "scanpatch/pii"
    assert entry["evaluation"]["policy_name"] == "external_default"
    assert entry["evaluation"]["runtime_mode"] == "cpu"
