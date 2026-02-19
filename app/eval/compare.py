from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _diff_num(a: Any, b: Any) -> float:
    try:
        return float(b) - float(a)
    except Exception:
        return 0.0


def _span_label_char_recalls(report: dict[str, Any]) -> dict[str, float]:
    span = (report.get("tasks") or {}).get("span_detection") or {}
    per_label = ((span.get("metrics") or {}).get("combined") or {}).get("per_label_char") or {}
    out: dict[str, float] = {}
    if isinstance(per_label, dict):
        for label, payload in per_label.items():
            if isinstance(payload, dict):
                out[str(label)] = float(payload.get("recall", 0.0) or 0.0)
    return out


def _headline(report: dict[str, Any]) -> float:
    span = (report.get("tasks") or {}).get("span_detection") or {}
    headline = span.get("headline") or {}
    try:
        return float(headline.get("risk_weighted_char_recall", 0.0) or 0.0)
    except Exception:
        return 0.0


def build_diff(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    base_head = _headline(base)
    new_head = _headline(new)

    base_recalls = _span_label_char_recalls(base)
    new_recalls = _span_label_char_recalls(new)
    labels = sorted(set(base_recalls) | set(new_recalls))
    per_label = []
    for label in labels:
        a = base_recalls.get(label, 0.0)
        b = new_recalls.get(label, 0.0)
        per_label.append({"label": label, "base": a, "new": b, "delta": b - a})
    per_label_sorted = sorted(per_label, key=lambda x: x["delta"])

    return {
        "diff_version": "1.0",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "headline": {
            "base": base_head,
            "new": new_head,
            "delta": new_head - base_head,
        },
        "span_detection": {
            "per_label_char_recall": per_label,
            "worst_10": per_label_sorted[:10],
            "best_10": list(reversed(per_label_sorted[-10:])),
        },
    }


def render_diff_markdown(diff: dict[str, Any]) -> str:
    head = diff.get("headline") or {}
    lines = [
        "# Guardrails Eval Diff",
        "",
        f"- Headline (risk-weighted char recall) delta: `{float(head.get('delta', 0.0)):+.4f}` "
        f"(base={float(head.get('base', 0.0)):.4f}, new={float(head.get('new', 0.0)):.4f})",
        "",
        "## Worst Per-Label Char Recall Deltas",
        "",
    ]
    for item in (diff.get("span_detection") or {}).get("worst_10") or []:
        lines.append(
            f"- `{item['label']}`: {item['delta']:+.4f} (base={item['base']:.4f}, new={item['new']:.4f})"
        )
    lines.append("")
    lines.append("## Best Per-Label Char Recall Deltas")
    lines.append("")
    for item in (diff.get("span_detection") or {}).get("best_10") or []:
        lines.append(
            f"- `{item['label']}`: {item['delta']:+.4f} (base={item['base']:.4f}, new={item['new']:.4f})"
        )
    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two eval reports.")
    p.add_argument("--base", required=True, help="Path to base report.json")
    p.add_argument("--new", required=True, help="Path to new report.json")
    p.add_argument("--out", required=True, help="Output directory for diff.json/diff.md")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    base = _load(args.base)
    new = _load(args.new)
    diff = build_diff(base, new)

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "diff.json").write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "diff.md").write_text(render_diff_markdown(diff), encoding="utf-8")
    print(json.dumps({"diff_dir": str(out), "diff_json": str(out / "diff.json"), "diff_md": str(out / "diff.md")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

