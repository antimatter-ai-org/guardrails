from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from app.config import load_policy_config
from app.eval.datasets.registry import get_dataset_adapter
from app.eval.env import load_env_file
from app.eval.labels import canonicalize_prediction_label
from app.eval.metrics import evaluate_samples, match_counts
from app.eval.predictor import detect_with_policy
from app.eval.report import build_report_payload, write_report_files
from app.eval.types import EvalSpan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual guardrails evaluation on public datasets.")
    parser.add_argument("--dataset", default="scanpatch/pii-ner-corpus-synthetic-controlled")
    parser.add_argument("--split", default="test")
    parser.add_argument("--policy-path", default="configs/policy.yaml")
    parser.add_argument("--policy-name", default=None)
    parser.add_argument("--cache-dir", default=".eval_cache/hf")
    parser.add_argument("--output-dir", default="reports/evaluations")
    parser.add_argument("--env-file", default=".env.eval")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--errors-preview-limit", type=int, default=25)
    return parser.parse_args()


def _configure_hf_cache(cache_dir: str) -> None:
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_path / "datasets"))


def _dataset_slug(name: str) -> str:
    safe = []
    for char in name.lower():
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def _warm_up_detectors(detectors: list[object], timeout_seconds: float = 120.0) -> None:
    for detector in detectors:
        runtime = getattr(detector, "_runtime", None)
        if runtime is None:
            detect_fn = getattr(detector, "detect", None)
            if callable(detect_fn):
                detect_fn("warmup")
            continue

        # For local runtime we explicitly force a synchronous load to avoid
        # evaluating with missing GLiNER predictions during async warm-up.
        if getattr(runtime, "_model", None) is None and callable(getattr(runtime, "_load_model", None)):
            runtime._load_model()
            if getattr(runtime, "_model", None) is None:
                load_error = getattr(runtime, "_load_error", None)
                if load_error:
                    raise RuntimeError(f"detector warm-up failed: {load_error}")

        detect_fn = getattr(detector, "detect", None)
        if callable(detect_fn):
            detect_fn("warmup")

        loading_started = bool(getattr(runtime, "_loading_started", False))
        if not loading_started and getattr(runtime, "_model", None) is not None:
            continue

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if getattr(runtime, "_model", None) is not None:
                break
            load_error = getattr(runtime, "_load_error", None)
            if load_error:
                raise RuntimeError(f"detector warm-up failed: {load_error}")
            time.sleep(0.2)


def main() -> int:
    args = _parse_args()
    load_env_file(args.env_file)
    _configure_hf_cache(args.cache_dir)

    hf_token = os.getenv(args.hf_token_env)
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    # Load policy and detectors only after env file is applied.
    from app.detectors.factory import build_registry
    from app.settings import settings

    policy_config = load_policy_config(args.policy_path)
    policy_name = args.policy_name or policy_config.default_policy
    if policy_name not in policy_config.policies:
        raise KeyError(f"Policy '{policy_name}' not found in {args.policy_path}")
    policy = policy_config.policies[policy_name]

    registry = build_registry(policy_config.detector_definitions)
    detectors = []
    for detector_name in policy.detectors:
        detector = registry.get(detector_name)
        if detector is not None:
            detectors.append(detector)

    if not detectors:
        raise RuntimeError("No active detectors available for evaluation policy")
    _warm_up_detectors(detectors)

    adapter = get_dataset_adapter(args.dataset)
    samples = adapter.load_samples(
        split=args.split,
        cache_dir=args.cache_dir,
        hf_token=hf_token,
        max_samples=args.max_samples,
    )

    predictions_by_id: dict[str, list[EvalSpan]] = {}
    errors_preview: list[dict[str, object]] = []

    started = time.perf_counter()
    for sample in samples:
        detections = detect_with_policy(sample.text, policy=policy, detectors=detectors)
        predicted_spans = [
            EvalSpan(
                start=item.start,
                end=item.end,
                label=item.label,
                canonical_label=canonicalize_prediction_label(item.label),
                score=item.score,
                detector=item.detector,
            )
            for item in detections
        ]
        predictions_by_id[sample.sample_id] = predicted_spans

        if len(errors_preview) < args.errors_preview_limit:
            counts = match_counts(
                gold_spans=sample.gold_spans,
                predicted_spans=predicted_spans,
                require_label=False,
                allow_overlap=False,
            )
            if counts.false_positives > 0 or counts.false_negatives > 0:
                errors_preview.append(
                    {
                        "sample_id": sample.sample_id,
                        "text": sample.text[:500],
                        "gold_spans": [
                            {"start": span.start, "end": span.end, "label": span.label}
                            for span in sample.gold_spans
                        ],
                        "predicted_spans": [
                            {"start": span.start, "end": span.end, "label": span.label}
                            for span in predicted_spans
                        ],
                    }
                )

    elapsed_seconds = time.perf_counter() - started
    aggregate = evaluate_samples(samples, predictions_by_id)

    report_payload = build_report_payload(
        dataset_name=args.dataset,
        split=args.split,
        sample_count=len(samples),
        policy_name=policy_name,
        policy_path=str(Path(args.policy_path).resolve()),
        runtime_mode=settings.runtime_mode,
        elapsed_seconds=elapsed_seconds,
        aggregate=aggregate,
        errors_preview=errors_preview,
    )

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    report_slug = f"eval_{_dataset_slug(args.dataset)}_{args.split}_{timestamp}"
    json_path, md_path = write_report_files(report_payload, output_dir=args.output_dir, report_slug=report_slug)

    print(f"[ok] Dataset: {args.dataset} ({args.split})")
    print(f"[ok] Samples: {len(samples)}")
    print(f"[ok] Policy: {policy_name}")
    print(f"[ok] JSON report: {json_path}")
    print(f"[ok] Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
