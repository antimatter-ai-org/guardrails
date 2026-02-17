from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid report json: {p}")
    return payload


def _headline(report: dict[str, Any]) -> dict[str, float]:
    head = report.get("results", {}).get("suite_summary", {}).get("headline", {})
    if not isinstance(head, dict):
        return {}
    out: dict[str, float] = {}
    for k in ("risk_weighted_macro_char_recall", "risk_weighted_macro_char_residual_miss_ratio"):
        try:
            out[k] = float(head.get(k))
        except Exception:
            continue
    return out


def _by_label_char_recall(report: dict[str, Any]) -> dict[str, float]:
    by_label = report.get("results", {}).get("by_label", {})
    if not isinstance(by_label, dict):
        return {}
    out: dict[str, float] = {}
    for label, item in by_label.items():
        if not isinstance(item, dict):
            continue
        char = item.get("char_canonical", {})
        if not isinstance(char, dict):
            continue
        try:
            out[str(label)] = float(char.get("recall"))
        except Exception:
            continue
    return out


def compare_reports(*, base_report_path: str, new_report: dict[str, Any]) -> dict[str, Any]:
    base = _load_json(base_report_path)
    if str(base.get("report_version")) != "3.0":
        raise ValueError("comparison currently supports report_version 3.0 only")
    if str(new_report.get("report_version")) != "3.0":
        raise ValueError("comparison currently supports report_version 3.0 only")

    base_head = _headline(base)
    new_head = _headline(new_report)
    headline_delta: dict[str, float] = {}
    for k in sorted(set(base_head) | set(new_head)):
        if k in base_head and k in new_head:
            headline_delta[k] = round(new_head[k] - base_head[k], 6)

    base_lbl = _by_label_char_recall(base)
    new_lbl = _by_label_char_recall(new_report)
    label_delta: dict[str, float] = {}
    for k in sorted(set(base_lbl) | set(new_lbl)):
        if k in base_lbl and k in new_lbl:
            label_delta[k] = round(new_lbl[k] - base_lbl[k], 6)

    # Rank changes for quick human scanning.
    regressions = sorted(label_delta.items(), key=lambda kv: kv[1])[:10]
    improvements = sorted(label_delta.items(), key=lambda kv: -kv[1])[:10]

    return {
        "base_report_path": str(Path(base_report_path).expanduser().resolve()),
        "headline_delta": headline_delta,
        "label_char_recall_delta": label_delta,
        "top_regressions": [{"label": k, "delta": v} for k, v in regressions],
        "top_improvements": [{"label": k, "delta": v} for k, v in improvements],
    }

