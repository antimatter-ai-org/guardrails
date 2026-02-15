from __future__ import annotations

import types
from pathlib import Path

from app.eval import run as eval_run


def _metric(tp: int, fp: int, fn: int) -> dict:
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "residual_miss_ratio": 0.0,
    }


def test_aggregate_metrics_from_dataset_reports_sums_counts() -> None:
    reports = [
        {
            "metrics": {
                "exact_agnostic": _metric(1, 2, 3),
                "overlap_agnostic": _metric(2, 3, 4),
                "exact_canonical": _metric(3, 4, 5),
                "overlap_canonical": _metric(4, 5, 6),
                "char_canonical": _metric(10, 2, 5),
                "token_canonical": _metric(7, 1, 2),
                "per_label_exact": {"person": _metric(1, 0, 1)},
                "per_label_char": {"person": _metric(4, 1, 2)},
            }
        },
        {
            "metrics": {
                "exact_agnostic": _metric(10, 20, 30),
                "overlap_agnostic": _metric(20, 30, 40),
                "exact_canonical": _metric(30, 40, 50),
                "overlap_canonical": _metric(40, 50, 60),
                "char_canonical": _metric(100, 20, 50),
                "token_canonical": _metric(70, 10, 20),
                "per_label_exact": {"person": _metric(2, 1, 3)},
                "per_label_char": {"person": _metric(5, 2, 4)},
            }
        },
    ]

    aggregate = eval_run._aggregate_metrics_from_dataset_reports(reports)
    assert aggregate.exact_canonical.true_positives == 33
    assert aggregate.exact_canonical.false_positives == 44
    assert aggregate.exact_canonical.false_negatives == 55
    assert aggregate.char_canonical.true_positives == 110
    assert aggregate.token_canonical.false_negatives == 22
    assert aggregate.per_label_exact["person"].true_positives == 3
    assert aggregate.per_label_char["person"].false_negatives == 6


def test_aggregate_detector_breakdown_from_reports_sums_counts() -> None:
    reports = [
        {
            "detector_breakdown": {
                "a": {
                    "prediction_count": 3,
                    "canonical_prediction_count": 2,
                    "overlap_agnostic": _metric(1, 2, 3),
                    "overlap_canonical": _metric(4, 5, 6),
                }
            }
        },
        {
            "detector_breakdown": {
                "a": {
                    "prediction_count": 7,
                    "canonical_prediction_count": 5,
                    "overlap_agnostic": _metric(10, 20, 30),
                    "overlap_canonical": _metric(40, 50, 60),
                }
            }
        },
    ]

    out = eval_run._aggregate_detector_breakdown_from_reports(reports)
    assert out["a"]["prediction_count"] == 10
    assert out["a"]["canonical_prediction_count"] == 7
    assert out["a"]["overlap_agnostic"]["true_positives"] == 11
    assert out["a"]["overlap_canonical"]["false_negatives"] == 66


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    signature = {"x": 1}
    reports = [{"name": "d1", "split": "test", "sample_count": 10}]
    path = tmp_path / "cp.json"
    eval_run._write_checkpoint(path, signature=signature, dataset_reports=reports)
    loaded = eval_run._load_checkpoint(path)
    assert loaded is not None
    assert loaded["run_signature"] == signature
    assert loaded["dataset_reports"] == reports


def test_default_checkpoint_path_is_deterministic() -> None:
    signature = {"a": [1, 2, 3], "b": "x"}
    p1 = eval_run._default_checkpoint_path("reports/evaluations", signature)
    p2 = eval_run._default_checkpoint_path("reports/evaluations", signature)
    assert p1 == p2


def test_run_signature_contains_core_fields() -> None:
    args = types.SimpleNamespace(
        policy_path="configs/policy.yaml",
        split="test",
        mode="baseline",
        strict_split=False,
        synthetic_test_size=0.2,
        synthetic_split_seed=42,
        max_samples=10,
        cascade_threshold=0.15,
        cascade_heavy_recognizers="gliner_pii_multilingual",
    )
    signature = eval_run._run_signature(args, policy_name="external_default", selected_datasets=["d1"])
    assert signature["policy_name"] == "external_default"
    assert signature["datasets"] == ["d1"]
    assert signature["max_samples"] == 10
