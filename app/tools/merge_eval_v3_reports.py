from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.eval.types import MetricCounts
from app.eval_v3.metrics.aggregation import macro_over_labels
from app.eval_v3.metrics.risk_weighting import risk_weighted_char_recall


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_from_payload(payload: dict[str, Any]) -> MetricCounts:
    return MetricCounts(
        true_positives=int(payload.get("true_positives", 0)),
        false_positives=int(payload.get("false_positives", 0)),
        false_negatives=int(payload.get("false_negatives", 0)),
    )


def _metric_payload_from_counts(counts: MetricCounts) -> dict[str, Any]:
    return {
        "true_positives": counts.true_positives,
        "false_positives": counts.false_positives,
        "false_negatives": counts.false_negatives,
        "precision": round(counts.precision, 6),
        "recall": round(counts.recall, 6),
        "f1": round(counts.f1, 6),
        "residual_miss_ratio": round(1.0 - counts.recall, 6),
    }


def _sum_metric_payloads(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ca = _metric_from_payload(a)
    cb = _metric_from_payload(b)
    merged = MetricCounts(
        true_positives=ca.true_positives + cb.true_positives,
        false_positives=ca.false_positives + cb.false_positives,
        false_negatives=ca.false_negatives + cb.false_negatives,
    )
    return _metric_payload_from_counts(merged)


def _sum_simple_int_maps(a: dict[str, Any], b: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in set(a.keys()) | set(b.keys()):
        out[str(key)] = int(a.get(key, 0)) + int(b.get(key, 0))
    return dict(sorted(out.items()))


def _merge_per_label_metric_payloads(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in set(a.keys()) | set(b.keys()):
        pa = a.get(label) or {}
        pb = b.get(label) or {}
        out[str(label)] = _sum_metric_payloads(dict(pa), dict(pb))
    return dict(sorted(out.items()))


def _merge_span_combined_payload(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    metric_keys = [
        "exact_agnostic",
        "overlap_agnostic",
        "exact_canonical",
        "overlap_canonical",
        "char_canonical",
        "token_canonical",
    ]
    out: dict[str, Any] = {}
    for key in metric_keys:
        out[key] = _sum_metric_payloads(dict(a.get(key, {})), dict(b.get(key, {})))
    out["per_label_exact"] = _merge_per_label_metric_payloads(
        dict(a.get("per_label_exact", {})), dict(b.get("per_label_exact", {}))
    )
    out["per_label_char"] = _merge_per_label_metric_payloads(
        dict(a.get("per_label_char", {})), dict(b.get("per_label_char", {}))
    )
    return out


def _merge_detector_breakdown(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for detector in set(a.keys()) | set(b.keys()):
        da = dict(a.get(detector, {}) or {})
        db = dict(b.get(detector, {}) or {})
        merged = {
            "prediction_count": int(da.get("prediction_count", 0)) + int(db.get("prediction_count", 0)),
            "scored_prediction_count": int(da.get("scored_prediction_count", 0)) + int(db.get("scored_prediction_count", 0)),
            "overlap_canonical": _sum_metric_payloads(dict(da.get("overlap_canonical", {})), dict(db.get("overlap_canonical", {}))),
            "char_canonical": _sum_metric_payloads(dict(da.get("char_canonical", {})), dict(db.get("char_canonical", {}))),
        }
        out[str(detector)] = merged
    return dict(sorted(out.items()))


def _compute_headline_from_per_label_char(per_label_char: dict[str, Any]) -> dict[str, Any]:
    counts = {label: _metric_from_payload(payload) for label, payload in per_label_char.items()}
    headline = risk_weighted_char_recall(per_label_char=counts)
    return {
        "risk_weighted_char_recall": round(headline.value, 6),
        "labels_included": headline.labels_included,
        "total_weight": round(headline.total_weight, 6),
    }


def _compute_macro(per_label: dict[str, Any]) -> dict[str, Any]:
    counts = {label: _metric_from_payload(payload) for label, payload in per_label.items()}
    macro = macro_over_labels(counts)
    return {
        "precision": round(macro.precision, 6),
        "recall": round(macro.recall, 6),
        "f1": round(macro.f1, 6),
        "labels_included": int(macro.labels_included),
    }


def _merge_span_detection_task(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["elapsed_seconds"] = round(float(a.get("elapsed_seconds", 0.0)) + float(b.get("elapsed_seconds", 0.0)), 6)
    out["sample_count"] = int(a.get("sample_count", 0)) + int(b.get("sample_count", 0))

    a_metrics = dict((a.get("metrics") or {}).get("combined") or {})
    b_metrics = dict((b.get("metrics") or {}).get("combined") or {})
    combined = _merge_span_combined_payload(a_metrics, b_metrics)
    out["metrics"] = {
        "combined": combined,
        "datasets": list((a.get("metrics") or {}).get("datasets") or []) + list((b.get("metrics") or {}).get("datasets") or []),
    }

    # Recompute headline + macro from merged counts.
    out["headline"] = _compute_headline_from_per_label_char(dict(combined.get("per_label_char", {})))
    out["macro_over_labels"] = {
        "char": _compute_macro(dict(combined.get("per_label_char", {}))),
        "exact": _compute_macro(dict(combined.get("per_label_exact", {}))),
    }

    out["dataset_slices"] = {}  # cannot recompute without sample metadata
    out["detector_breakdown"] = _merge_detector_breakdown(
        dict(a.get("detector_breakdown", {})), dict(b.get("detector_breakdown", {}))
    )
    out["unscored_predictions"] = _sum_simple_int_maps(
        dict(a.get("unscored_predictions", {})), dict(b.get("unscored_predictions", {}))
    )
    return out


def _merge_policy_action_task(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["elapsed_seconds"] = round(float(a.get("elapsed_seconds", 0.0)) + float(b.get("elapsed_seconds", 0.0)), 6)
    out["sample_count"] = max(int(a.get("sample_count", 0)), int(b.get("sample_count", 0)))

    policies: dict[str, Any] = {}
    pa = dict(a.get("policies", {}) or {})
    pb = dict(b.get("policies", {}) or {})
    for policy_name in set(pa.keys()) | set(pb.keys()):
        ra = dict((pa.get(policy_name) or {}).get("metrics") or {})
        rb = dict((pb.get(policy_name) or {}).get("metrics") or {})
        tp = int(ra.get("tp", 0)) + int(rb.get("tp", 0))
        fp = int(ra.get("fp", 0)) + int(rb.get("fp", 0))
        tn = int(ra.get("tn", 0)) + int(rb.get("tn", 0))
        fn = int(ra.get("fn", 0)) + int(rb.get("fn", 0))

        # Derive metrics.
        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
        fnr = 0.0 if (fn + tp) == 0 else fn / (fn + tp)

        positive_action = (pa.get(policy_name) or pb.get(policy_name) or {}).get("positive_action") or "MASKED"
        policies[str(policy_name)] = {
            "positive_action": positive_action,
            "metrics": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "false_positive_rate": round(fpr, 6),
                "false_negative_rate": round(fnr, 6),
            },
        }

    out["policies"] = dict(sorted(policies.items()))
    return out


def _merge_mask_leakage_task(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    total = int(a.get("total_gold_spans", 0)) + int(b.get("total_gold_spans", 0))
    leaked = int(a.get("leaked_gold_spans", 0)) + int(b.get("leaked_gold_spans", 0))
    leakage = 0.0 if total == 0 else leaked / total
    return {
        "elapsed_seconds": round(float(a.get("elapsed_seconds", 0.0)) + float(b.get("elapsed_seconds", 0.0)), 6),
        "sample_count": int(a.get("sample_count", 0)) + int(b.get("sample_count", 0)),
        "processed_samples": int(a.get("processed_samples", 0)) + int(b.get("processed_samples", 0)),
        "total_gold_spans": total,
        "leaked_gold_spans": leaked,
        "leakage_fraction": round(leakage, 6),
        "leaked_examples": list(a.get("leaked_examples", []) or []) + list(b.get("leaked_examples", []) or []),
    }


def merge_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("need at least two reports to merge")
    base = reports[0]
    for other in reports[1:]:
        if base.get("report_version") != other.get("report_version"):
            raise ValueError("report_version mismatch")
        if base.get("run", {}).get("suite") != other.get("run", {}).get("suite"):
            raise ValueError("suite mismatch")
        if base.get("run", {}).get("split") != other.get("run", {}).get("split"):
            raise ValueError("split mismatch")
        if base.get("run", {}).get("policy_name") != other.get("run", {}).get("policy_name"):
            raise ValueError("policy_name mismatch")

    merged: dict[str, Any] = {
        "report_version": base.get("report_version"),
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run": dict(base.get("run") or {}),
        "tasks": {},
    }
    merged["run"]["run_id"] = f"merged_evalv3_{merged['run'].get('suite')}_{merged['run'].get('split')}_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}"
    merged["run"]["merged_from_reports"] = [r.get("run", {}).get("run_id") for r in reports]
    merged["run"]["datasets"] = sorted({ds for r in reports for ds in (r.get("run", {}).get("datasets") or [])})

    # Merge timing: best-effort sum across reports.
    timing_keys = ["dataset_load_seconds", "span_detection_seconds", "policy_action_seconds", "mask_leakage_seconds", "wall_seconds"]
    timing: dict[str, float] = {}
    for key in timing_keys:
        timing[key] = round(sum(float(r.get("run", {}).get("timing", {}).get(key, 0.0) or 0.0) for r in reports), 6)
    merged["run"]["timing"] = timing

    # Merge tasks pairwise.
    tasks = ["span_detection", "policy_action", "mask_leakage"]
    for task_name in tasks:
        present = [r.get("tasks", {}).get(task_name) for r in reports if (r.get("tasks", {}) or {}).get(task_name) is not None]
        if not present:
            continue
        current = present[0]
        for nxt in present[1:]:
            if task_name == "span_detection":
                current = _merge_span_detection_task(dict(current), dict(nxt))
            elif task_name == "policy_action":
                current = _merge_policy_action_task(dict(current), dict(nxt))
            elif task_name == "mask_leakage":
                current = _merge_mask_leakage_task(dict(current), dict(nxt))
        merged["tasks"][task_name] = current

    merged["run"]["finished_at_utc"] = datetime.now(tz=UTC).isoformat()
    return merged


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge multiple eval_v3 report.json files into a single aggregate report.")
    p.add_argument("--out", required=True, help="Output path for merged report.json")
    p.add_argument("reports", nargs="+", help="Input report.json paths")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    reports = [_read_json(path) for path in args.reports]
    merged = merge_reports(reports)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

