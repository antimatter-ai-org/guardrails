from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Iterable

from app.eval_v3.config import load_eval_registry
from app.eval_v3.datasets.hf_span_dataset import build_samples_from_hf_split, load_hf_split


SAFE_WORDS = [
    "data",
    "system",
    "process",
    "request",
    "response",
    "service",
    "signal",
    "value",
    "example",
    "context",
    "sample",
    "message",
    "stream",
    "buffer",
    "cache",
    "window",
    "token",
    "analysis",
    "pipeline",
    "runtime",
    "result",
    "metric",
    "segment",
    "record",
    "document",
    "source",
    "target",
    "worker",
    "thread",
    "queue",
    "trace",
    "event",
    "profile",
    "label",
    "field",
    "lookup",
    "policy",
    "engine",
    "payload",
    "storage",
    "cluster",
    "endpoint",
    "region",
    "memory",
    "vector",
    "signal",
    "format",
    "report",
    "status",
    "batch",
    "sample",
    "content",
    "context",
    "update",
    "monitor",
    "router",
    "gateway",
    "module",
    "command",
    "config",
    "schema",
    "version",
    "selector",
    "record",
    "index",
    "output",
    "input",
    "graph",
    "feature",
    "value",
    "history",
    "log",
    "state",
    "range",
    "matrix",
    "vector",
    "kernel",
    "frame",
    "node",
    "cache",
    "stream",
    "packet",
    "chunk",
    "buffer",
    "timer",
    "clock",
    "signal",
    "query",
    "response",
    "filter",
    "sample",
    "series",
    "trace",
    "control",
    "layout",
    "diagram",
    "report",
    "store",
    "index",
    "compute",
    "model",
    "engine",
    "token",
    "result",
    "session",
    "client",
    "server",
    "worker",
]

DEFAULT_SOURCE_DATASETS = [
    "antimatter-ai/guardrails-ru-russian-pii-66k-mvp-v1",
    "antimatter-ai/guardrails-ru-meddies-pii-cleaned-v1",
    "antimatter-ai/guardrails-ru-scanpatch-pii-ner-controlled-v1",
    "antimatter-ai/guardrails-ru-rubai-ner-150k-personal-ru-v1",
    "antimatter-ai/guardrails-ru-kaggle-pii-data-detection-sample-v1",
    "antimatter-ai/guardrails-ru-hf-pii-sample-en-v1",
    "antimatter-ai/guardrails-ru-presidio-test-dataset-v1",
]

LABELS = [
    "person",
    "email",
    "phone",
    "ip",
    "url",
    "identifier",
    "date",
    "location",
    "secret",
    "payment_card",
    "organization",
]

BUCKETS = [
    ("10k", 10_000),
    ("50k", 50_000),
    ("100k", 100_000),
    ("250k", 250_000),
    ("1m", 1_000_000),
]


@dataclass(frozen=True)
class Segment:
    text: str
    spans: list[dict[str, int | str]]
    language: str | None
    script_profile: str | None
    source_dataset: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate long-context ablation datasets (pad or concat).")
    p.add_argument("--mode", choices=["pad", "concat"], required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="fast")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-bucket", type=int, default=16)
    p.add_argument(
        "--bucket",
        action="append",
        default=None,
        help="Override bucket list using label=length (e.g., 10k=10000).",
    )
    p.add_argument("--registry-path", default=str(Path("configs") / "eval" / "suites.yaml"))
    p.add_argument("--cache-dir", default=str(Path(".eval_cache") / "hf"))
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    p.add_argument("--source-dataset", action="append", default=None)
    return p.parse_args()


def _parse_buckets(raw: list[str] | None) -> list[tuple[str, int]]:
    if not raw:
        return BUCKETS
    parsed: list[tuple[str, int]] = []
    for item in raw:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"bucket override must be label=length, got '{item}'")
        label, raw_size = item.split("=", 1)
        label = label.strip()
        size = int(raw_size.strip())
        if not label:
            raise ValueError(f"bucket label missing in '{item}'")
        if size <= 0:
            raise ValueError(f"bucket size must be positive, got {size} for '{item}'")
        parsed.append((label, size))
    if not parsed:
        raise ValueError("no valid buckets provided")
    return parsed


def _make_filler(target_len: int, rng: random.Random) -> str:
    if target_len <= 0:
        return ""
    words: list[str] = []
    while len(words) < max(8, target_len // 6):
        words.append(rng.choice(SAFE_WORDS))
    parts: list[str] = []
    for idx, word in enumerate(words, start=1):
        parts.append(word)
        if idx % 12 == 0:
            parts.append(".")
    text = " ".join(parts)
    if len(text) < target_len:
        # Repeat to reach target length deterministically.
        repeats = (target_len // max(1, len(text))) + 1
        text = (text + " ") * repeats
    return text[:target_len]


def _load_segments(*, dataset_ids: Iterable[str], split: str, registry_path: str, cache_dir: str, hf_token_env: str) -> list[Segment]:
    registry = load_eval_registry(Path(registry_path))
    hf_token = None
    env_name = str(hf_token_env)
    if env_name:
        from os import getenv
        hf_token = getenv(env_name) or None

    segments: list[Segment] = []
    for dataset_id in dataset_ids:
        cfg = registry.datasets[dataset_id]
        ds, _ = load_hf_split(hf_id=cfg.hf_id, split=split, cache_dir=cache_dir, hf_token=hf_token)
        samples = build_samples_from_hf_split(
            dataset_id=dataset_id,
            split=split,
            ds=ds,
            text_field=cfg.text_field,
            spans_field=cfg.spans_field,
            label_map=cfg.label_map,
            slice_fields=cfg.slice_fields,
            selected_indices=None,
            max_samples=None,
        )
        for sample in samples:
            text = str(sample.text or "")
            if not text:
                continue
            spans: list[dict[str, int | str]] = []
            for sp in sample.gold_spans:
                label = sp.canonical_label
                if not label or label not in LABELS:
                    continue
                if sp.end <= sp.start:
                    continue
                spans.append({"start": int(sp.start), "end": int(sp.end), "label": str(label)})
            segments.append(
                Segment(
                    text=text,
                    spans=spans,
                    language=str(sample.metadata.get("language")) if sample.metadata.get("language") is not None else None,
                    script_profile=(
                        str(sample.metadata.get("script_profile"))
                        if sample.metadata.get("script_profile") is not None
                        else None
                    ),
                    source_dataset=dataset_id,
                )
            )
    return segments


def _choose_language(values: list[str | None]) -> str:
    cleaned = {v for v in values if v}
    if len(cleaned) == 1:
        return next(iter(cleaned))
    if not cleaned:
        return "unknown"
    return "mixed"


def _sorted_spans(spans: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    return sorted(spans, key=lambda s: (int(s["start"]), int(s["end"])))


def _build_pad_sample(
    *,
    base: Segment,
    target_len: int,
    placement: str,
    rng: random.Random,
) -> tuple[str, list[dict[str, int | str]], str]:
    sep = "\n\n"
    sep_len = len(sep)
    base_text = base.text
    if len(base_text) + sep_len > target_len:
        raise ValueError("base too long for target")

    if placement == "middle":
        filler_len = target_len - len(base_text) - sep_len * 2
    else:
        filler_len = target_len - len(base_text) - sep_len
    if filler_len < 0:
        raise ValueError("insufficient filler length")

    filler = _make_filler(filler_len, rng)

    pieces: list[tuple[str, list[dict[str, int | str]]]] = []
    if placement == "head":
        pieces = [(base_text, base.spans), (sep, []), (filler, [])]
    elif placement == "tail":
        pieces = [(filler, []), (sep, []), (base_text, base.spans)]
    else:
        pre_len = filler_len // 2
        post_len = filler_len - pre_len
        filler_pre = filler[:pre_len]
        filler_post = filler[pre_len:pre_len + post_len]
        pieces = [(filler_pre, []), (sep, []), (base_text, base.spans), (sep, []), (filler_post, [])]

    text_parts: list[str] = []
    spans: list[dict[str, int | str]] = []
    offset = 0
    for chunk, chunk_spans in pieces:
        if chunk:
            text_parts.append(chunk)
            for sp in chunk_spans:
                spans.append(
                    {
                        "start": int(sp["start"]) + offset,
                        "end": int(sp["end"]) + offset,
                        "label": sp["label"],
                    }
                )
            offset += len(chunk)
    text = "".join(text_parts)
    return text, _sorted_spans(spans), placement


def _build_concat_sample(
    *,
    segments: list[Segment],
    target_len: int,
    rng: random.Random,
    max_overrun: float,
) -> tuple[str, list[dict[str, int | str]], list[Segment]]:
    sep = "\n\n"
    sep_len = len(sep)
    pieces: list[tuple[str, list[dict[str, int | str]]]] = []
    used: list[Segment] = []
    total = 0

    while total < target_len:
        seg = rng.choice(segments)
        if pieces:
            pieces.append((sep, []))
            total += sep_len
        pieces.append((seg.text, seg.spans))
        total += len(seg.text)
        used.append(seg)
        if total > int(target_len * max_overrun):
            raise ValueError("overrun")

    text_parts: list[str] = []
    spans: list[dict[str, int | str]] = []
    offset = 0
    for chunk, chunk_spans in pieces:
        if chunk:
            text_parts.append(chunk)
            for sp in chunk_spans:
                spans.append(
                    {
                        "start": int(sp["start"]) + offset,
                        "end": int(sp["end"]) + offset,
                        "label": sp["label"],
                    }
                )
            offset += len(chunk)
    text = "".join(text_parts)
    return text, _sorted_spans(spans), used


def main() -> None:
    args = _parse_args()
    rng = random.Random(int(args.seed))
    buckets = _parse_buckets(args.bucket)

    dataset_ids = args.source_dataset or DEFAULT_SOURCE_DATASETS
    segments = _load_segments(
        dataset_ids=dataset_ids,
        split=args.split,
        registry_path=str(args.registry_path),
        cache_dir=str(args.cache_dir),
        hf_token_env=str(args.hf_token_env),
    )
    if not segments:
        raise RuntimeError("no source segments loaded")

    rows: list[dict[str, object]] = []
    for bucket_label, target_len in buckets:
        for idx in range(int(args.samples_per_bucket)):
            for attempt in range(50):
                try:
                    if args.mode == "pad":
                        base = rng.choice(segments)
                        if len(base.text) >= target_len:
                            continue
                        placement = rng.choice(["head", "middle", "tail"])
                        text, spans, placement_profile = _build_pad_sample(
                            base=base,
                            target_len=target_len,
                            placement=placement,
                            rng=rng,
                        )
                        language = base.language or "unknown"
                        script_profile = base.script_profile or "unknown"
                        source_datasets = base.source_dataset
                        source = "pad_short"
                        fmt = "narrative_pad"
                        noisy = False
                    else:
                        max_overrun = 1.15 if target_len <= 50_000 else 1.08
                        text, spans, used_segments = _build_concat_sample(
                            segments=segments,
                            target_len=target_len,
                            rng=rng,
                            max_overrun=max_overrun,
                        )
                        langs = [seg.language for seg in used_segments]
                        scripts = [seg.script_profile for seg in used_segments]
                        language = _choose_language(langs)
                        script_profile = _choose_language(scripts)
                        source_datasets = ",".join(sorted({seg.source_dataset for seg in used_segments}))
                        source = "concat_short"
                        fmt = "narrative_concat"
                        placement_profile = "spread"
                        noisy = False

                    row = {
                        "id": f"{args.mode}-{bucket_label}-{idx}",
                        "source_uid": f"{args.mode}-{bucket_label}-{idx}",
                        "source_text": text,
                        "privacy_mask": spans,
                        "language": language,
                        "script_profile": script_profile,
                        "format": fmt,
                        "length_bucket": bucket_label,
                        "placement_profile": placement_profile,
                        "entity_count": len(spans),
                        "length_chars": len(text),
                        "source": source,
                        "noisy": noisy,
                        "source_datasets": source_datasets,
                    }
                    rows.append(row)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError(f"unable to build sample for {bucket_label} after retries")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import Dataset, DatasetDict  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets package is required; install guardrails-service[eval]") from exc

    ds = Dataset.from_list(rows)
    dd = DatasetDict({args.split: ds})
    dd.save_to_disk(str(output_dir))

    meta = {
        "mode": args.mode,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "source_datasets": list(dataset_ids),
        "split": args.split,
        "samples_per_bucket": int(args.samples_per_bucket),
        "buckets": {name: size for name, size in buckets},
        "total_samples": len(rows),
    }
    (output_dir / "generation_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
