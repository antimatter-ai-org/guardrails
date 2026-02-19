from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import load_policy_config
from app.core.analysis.service import PresidioAnalysisService
from app.eval.predictors.analyze_text import as_eval_spans
from app.model_assets import apply_model_env
from app.runtime.tokenizer_chunking import chunk_text, deterministic_overlap_tokens, effective_max_tokens_for_token_classifier
from app.settings import settings


@dataclass(frozen=True)
class InsertCase:
    name: str
    offset: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose chunk-boundary behavior for long text.")
    p.add_argument("--policy-path", default="configs/policy.yaml")
    p.add_argument("--policy-name", default="external")
    p.add_argument("--model-name", default=None, help="Override HF model name for tokenizer config.")
    p.add_argument("--label", default="email")
    p.add_argument("--output", required=True)
    p.add_argument("--min-length", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _ensure_runtime_ready(*, service: PresidioAnalysisService, analyzer_profile: str, policy_name: str) -> None:
    errors = service.ensure_profile_runtimes_ready(
        profile_names=[analyzer_profile],
        timeout_s=settings.pytriton_init_timeout_s,
    )
    if errors:
        raise RuntimeError(f"model runtime readiness check failed for policy '{policy_name}': {errors}")


def _build_base_text(min_length: int) -> str:
    chunk = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 50).strip()
    text = chunk
    while len(text) < min_length:
        text = text + "\n" + chunk
    return text


def _insert(text: str, offset: int, marker: str) -> tuple[str, int, int]:
    offset = max(0, min(len(text), offset))
    new_text = text[:offset] + marker + text[offset:]
    return new_text, offset, offset + len(marker)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return min(a_end, b_end) > max(a_start, b_start)


def main() -> None:
    args = _parse_args()
    apply_model_env(settings.model_dir, settings.offline_mode)

    policy_cfg = load_policy_config(Path(args.policy_path))
    if args.policy_name not in policy_cfg.policies:
        raise RuntimeError(f"unknown policy: {args.policy_name}")
    policy = policy_cfg.policies[args.policy_name]

    model_name = args.model_name
    if model_name is None:
        # Try policy recognizer config first.
        recognizer = policy_cfg.recognizer_definitions.get("nemotron_pii_token_classifier")
        if recognizer:
            params = getattr(recognizer, "params", {}) or {}
            model_name = str(params.get("model_name") or "")
    if not model_name:
        model_name = "scanpatch/pii-ner-nemotron"

    from transformers import AutoConfig, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name)

    class Dummy:
        pass

    dummy = Dummy()
    dummy.config = config
    max_tokens = effective_max_tokens_for_token_classifier(model=dummy, tokenizer=tokenizer)
    overlap_tokens = deterministic_overlap_tokens(max_tokens)

    base = _build_base_text(int(args.min_length))
    windows = chunk_text(text=base, tokenizer=tokenizer, max_input_tokens=max_tokens, overlap_tokens=overlap_tokens)
    if len(windows) < 2:
        base = _build_base_text(int(args.min_length) * 2)
        windows = chunk_text(text=base, tokenizer=tokenizer, max_input_tokens=max_tokens, overlap_tokens=overlap_tokens)
    if len(windows) < 2:
        raise RuntimeError("unable to produce multiple chunk windows for diagnostic")

    # Choose a boundary in the middle to avoid edge effects.
    boundary = windows[1].text_start
    marker = " test.user@example.com "

    cases = [
        InsertCase(name="cross_boundary", offset=max(0, boundary - len(marker) // 2)),
        InsertCase(name="just_before", offset=max(0, boundary - len(marker) - 2)),
        InsertCase(name="just_after", offset=boundary + 2),
    ]

    service = PresidioAnalysisService(policy_cfg)
    _ensure_runtime_ready(service=service, analyzer_profile=policy.analyzer_profile, policy_name=args.policy_name)

    results: list[dict[str, Any]] = []
    for case in cases:
        text, start, end = _insert(base, case.offset, marker)
        preds = service.analyze_text(text=text, profile_name=policy.analyzer_profile, policy_min_score=float(policy.min_score))
        spans = [sp for sp in as_eval_spans(preds) if sp.canonical_label == args.label]
        overlaps = [sp for sp in spans if _overlaps(int(sp.start), int(sp.end), start, end)]
        results.append(
            {
                "case": case.name,
                "expected": {"start": start, "end": end, "len": end - start},
                "predicted_count": len(spans),
                "overlap_count": len(overlaps),
                "overlaps": [
                    {
                        "start": int(sp.start),
                        "end": int(sp.end),
                        "len": int(sp.end) - int(sp.start),
                        "span_inflation": (int(sp.end) - int(sp.start)) - (end - start),
                    }
                    for sp in overlaps
                ],
            }
        )

    payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "policy_name": args.policy_name,
        "model_name": model_name,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "window_count": len(windows),
        "boundary": boundary,
        "label": args.label,
        "results": results,
    }

    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
