from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class GateResult:
    enforced: bool
    failures: tuple[str, ...]


def load_gates(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {}
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def evaluate_gates(
    *,
    gates: dict[str, Any],
    report: dict[str, Any],
    enforce: bool,
) -> GateResult:
    if not gates:
        return GateResult(enforced=False, failures=tuple())

    failures: list[str] = []

    suite_gates = gates.get("suite") or {}
    summ = report["results"]["suite_summary"]["headline"]
    if isinstance(suite_gates, dict):
        min_recall = suite_gates.get("headline_char_recall_min")
        if min_recall is not None and float(summ["risk_weighted_macro_char_recall"]) < float(min_recall):
            failures.append(
                f"suite headline_char_recall {summ['risk_weighted_macro_char_recall']} < {float(min_recall):.6f}"
            )
        max_resid = suite_gates.get("headline_char_residual_miss_ratio_max")
        if max_resid is not None and float(summ["risk_weighted_macro_char_residual_miss_ratio"]) > float(max_resid):
            failures.append(
                f"suite headline_char_residual {summ['risk_weighted_macro_char_residual_miss_ratio']} > {float(max_resid):.6f}"
            )

    per_label = gates.get("per_label") or {}
    if isinstance(per_label, dict):
        by_label = report["results"]["by_label"]
        for label, constraints in per_label.items():
            if not isinstance(constraints, dict):
                continue
            if label not in by_label:
                continue
            char_recall = float(by_label[label]["char_canonical"]["recall"])
            char_min = constraints.get("char_recall_min")
            if char_min is not None and char_recall < float(char_min):
                failures.append(f"label {label} char_recall {char_recall:.6f} < {float(char_min):.6f}")

    per_dataset = gates.get("per_dataset") or {}
    if isinstance(per_dataset, dict):
        by_ds = {d["dataset_id"]: d for d in report["results"]["by_dataset"]}
        for ds_id, ds_constraints in per_dataset.items():
            if not isinstance(ds_constraints, dict):
                continue
            ds_item = by_ds.get(ds_id)
            if ds_item is None:
                continue
            ds_labels = ds_item["metrics"]["per_label_char"]
            for label, constraints in ds_constraints.items():
                if not isinstance(constraints, dict):
                    continue
                if label not in ds_labels:
                    continue
                recall = float(ds_labels[label]["recall"])
                min_recall = constraints.get("char_recall_min")
                if min_recall is not None and recall < float(min_recall):
                    failures.append(f"dataset {ds_id} label {label} char_recall {recall:.6f} < {float(min_recall):.6f}")

    if not enforce:
        # Not enforced: still report failures as warnings, but no failing exit.
        return GateResult(enforced=False, failures=tuple(failures))
    return GateResult(enforced=True, failures=tuple(failures))

