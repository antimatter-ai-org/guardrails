from __future__ import annotations

from typing import Any


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_report_markdown(report: dict[str, Any]) -> str:
    run = report.get("run", {})
    suite = run.get("suite", "unknown")
    split = run.get("split", "unknown")
    tasks = run.get("tasks", [])
    runtime = run.get("runtime", {})
    timing = run.get("timing", {}) if isinstance(run.get("timing"), dict) else {}
    settings = run.get("settings", {}) if isinstance(run.get("settings"), dict) else {}

    lines: list[str] = [
        "# Guardrails Evaluation Report (v3)",
        "",
        f"- Suite: `{suite}`",
        f"- Split: `{split}`",
        f"- Datasets: `{len(run.get('datasets', []) or [])}`",
        f"- Tasks: `{', '.join(tasks)}`",
        f"- Runtime: `{runtime.get('mode', 'cpu')}` (cpu_device=`{runtime.get('cpu_device', 'auto')}`)",
        f"- Nemotron: `{bool(settings.get('enable_nemotron', False))}`",
        f"- Offline: `{bool(run.get('offline', False))}`",
        f"- Generated: `{report.get('generated_at_utc', '')}`",
        f"- Wall time: `{float(timing.get('wall_seconds', 0.0) or 0.0):.2f}s` "
        f"(load={float(timing.get('dataset_load_seconds', 0.0) or 0.0):.2f}s, "
        f"span={float(timing.get('span_detection_seconds', 0.0) or 0.0):.2f}s, "
        f"action={float(timing.get('policy_action_seconds', 0.0) or 0.0):.2f}s, "
        f"leak={float(timing.get('mask_leakage_seconds', 0.0) or 0.0):.2f}s)",
        "",
    ]

    span = (report.get("tasks") or {}).get("span_detection")
    if isinstance(span, dict):
        headline = span.get("headline", {})
        lines.extend(
            [
                "## Span Detection",
                "",
                f"- Headline (risk-weighted char recall): `{_fmt_pct(float(headline.get('risk_weighted_char_recall', 0.0)))}` "
                f"(labels_included={headline.get('labels_included', 0)})",
            ]
        )
        combined = (span.get("metrics") or {}).get("combined") or {}
        exact = (combined.get("exact_canonical") or {})
        overlap = (combined.get("overlap_canonical") or {})
        char = (combined.get("char_canonical") or {})
        token = (combined.get("token_canonical") or {})
        lines.extend(
            [
                f"- overlap_canonical: P={overlap.get('precision', 0):.4f} R={overlap.get('recall', 0):.4f} F1={overlap.get('f1', 0):.4f}",
                f"- char_canonical: P={char.get('precision', 0):.4f} R={char.get('recall', 0):.4f} F1={char.get('f1', 0):.4f}",
                f"- token_canonical: P={token.get('precision', 0):.4f} R={token.get('recall', 0):.4f} F1={token.get('f1', 0):.4f}",
                f"- exact_canonical: P={exact.get('precision', 0):.4f} R={exact.get('recall', 0):.4f} F1={exact.get('f1', 0):.4f}",
                "",
            ]
        )

        datasets = ((span.get("metrics") or {}).get("datasets")) or []
        if isinstance(datasets, list) and datasets:
            lines.extend(["### Per-Dataset Runtime (span_detection)", ""])
            lines.append("| dataset | rows | seconds | rows/s |")
            lines.append("|---|---:|---:|---:|")
            for item in sorted(datasets, key=lambda d: float((d or {}).get("elapsed_seconds", 0.0) or 0.0), reverse=True):
                if not isinstance(item, dict):
                    continue
                ds_id = item.get("dataset_id", "")
                rows = int(item.get("sample_count", 0) or 0)
                sec = float(item.get("elapsed_seconds", 0.0) or 0.0)
                rps = float(item.get("samples_per_second", 0.0) or 0.0)
                lines.append(f"| `{ds_id}` | {rows} | {sec:.2f} | {rps:.2f} |")
            lines.append("")

        macro = span.get("macro_over_labels", {})
        if isinstance(macro, dict) and macro:
            lines.extend(
                [
                    "### Macro Over Labels (Gold-Supported)",
                    "",
                    f"- char: P={macro.get('char', {}).get('precision', 0):.4f} "
                    f"R={macro.get('char', {}).get('recall', 0):.4f} "
                    f"F1={macro.get('char', {}).get('f1', 0):.4f} "
                    f"(labels={macro.get('char', {}).get('labels_included', 0)})",
                    "",
                ]
            )

    action = (report.get("tasks") or {}).get("policy_action")
    if isinstance(action, dict):
        lines.extend(["## Policy Action", ""])
        if "elapsed_seconds" in action:
            lines.append(f"- Elapsed: `{float(action.get('elapsed_seconds', 0.0) or 0.0):.2f}s`")
            lines.append("")
        policies = action.get("policies", {})
        if isinstance(policies, dict):
            for policy_name, payload in policies.items():
                metrics = (payload or {}).get("metrics") or {}
                lines.append(f"### `{policy_name}`")
                lines.append("")
                lines.append(
                    f"- P={metrics.get('precision', 0):.4f} R={metrics.get('recall', 0):.4f} F1={metrics.get('f1', 0):.4f} "
                    f"FNR={metrics.get('false_negative_rate', 0):.4f} FPR={metrics.get('false_positive_rate', 0):.4f}"
                )
                lines.append("")

    leakage = (report.get("tasks") or {}).get("mask_leakage")
    if isinstance(leakage, dict):
        lines.extend(["## Mask Leakage (Diagnostic)", ""])
        frac = float(leakage.get("leakage_fraction", 0.0) or 0.0)
        if "elapsed_seconds" in leakage:
            lines.append(f"- Elapsed: `{float(leakage.get('elapsed_seconds', 0.0) or 0.0):.2f}s`")
        lines.append(f"- Gold spans leaked verbatim (approx): `{_fmt_pct(frac)}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
