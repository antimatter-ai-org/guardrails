from __future__ import annotations

from app.tools.merge_eval_reports import merge_reports


def _minimal_metric(tp: int, fp: int, fn: int) -> dict:
    # precision/recall/f1 are recomputed by the merger; but we provide placeholders.
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "residual_miss_ratio": 1.0,
    }


def _minimal_report(*, run_id: str, datasets: list[str], tp: int) -> dict:
    combined = {
        "exact_agnostic": _minimal_metric(tp, 0, 0),
        "overlap_agnostic": _minimal_metric(tp, 0, 0),
        "exact_canonical": _minimal_metric(tp, 0, 0),
        "overlap_canonical": _minimal_metric(tp, 0, 0),
        "char_canonical": _minimal_metric(tp, 0, 0),
        "token_canonical": _minimal_metric(tp, 0, 0),
        "per_label_exact": {"person": _minimal_metric(tp, 0, 0)},
        "per_label_char": {"person": _minimal_metric(tp, 0, 0)},
    }
    return {
        "report_version": "3.0",
        "generated_at_utc": "2026-01-01T00:00:00Z",
        "run": {
            "run_id": run_id,
            "suite": "guardrails_ru",
            "split": "fast",
            "policy_name": "external",
            "datasets": datasets,
            "timing": {},
        },
        "tasks": {
            "span_detection": {
                "elapsed_seconds": 1.0,
                "sample_count": 1,
                "headline": {},
                "metrics": {"combined": combined, "datasets": []},
                "macro_over_labels": {},
                "dataset_slices": {},
                "detector_breakdown": {},
                "unscored_predictions": {},
            },
            "policy_action": {
                "elapsed_seconds": 1.0,
                "sample_count": 1,
                "policies": {
                    "external": {
                        "positive_action": "MASKED",
                        "metrics": {"tp": tp, "fp": 0, "tn": 0, "fn": 0},
                    }
                },
            },
            "mask_leakage": {
                "elapsed_seconds": 1.0,
                "sample_count": 1,
                "processed_samples": 1,
                "total_gold_spans": tp,
                "leaked_gold_spans": 0,
                "leakage_fraction": 0.0,
                "leaked_examples": [],
            },
        },
    }


def test_merge_reports_sums_counts_and_unions_datasets() -> None:
    a = _minimal_report(run_id="a", datasets=["d1"], tp=2)
    b = _minimal_report(run_id="b", datasets=["d2"], tp=3)
    a["tasks"]["span_detection"]["unscored_predictions"] = {"d1": {"person": 1}}
    b["tasks"]["span_detection"]["unscored_predictions"] = {"d2": {"person": 2, "email": 1}}
    merged = merge_reports([a, b])

    assert merged["run"]["datasets"] == ["d1", "d2"]
    span = merged["tasks"]["span_detection"]["metrics"]["combined"]
    assert span["char_canonical"]["true_positives"] == 5
    assert merged["tasks"]["policy_action"]["policies"]["external"]["metrics"]["tp"] == 5
    assert merged["tasks"]["mask_leakage"]["total_gold_spans"] == 5
    assert merged["tasks"]["span_detection"]["unscored_predictions"] == {
        "d1": {"person": 1},
        "d2": {"email": 1, "person": 2},
    }
