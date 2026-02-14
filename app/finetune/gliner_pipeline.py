from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.eval.datasets.registry import get_dataset_adapter
from app.eval.labels import canonicalize_prediction_label
from app.eval.metrics import EvaluationAggregate, evaluate_samples
from app.eval.types import EvalSample, EvalSpan
from app.runtime.torch_runtime import resolve_torch_device

DEFAULT_SCANPATCH_DATASET = "scanpatch/pii-ner-corpus-synthetic-controlled"
DEFAULT_TARGET_LABELS = [
    "person",
    "organization",
    "location",
    "email",
    "phone",
    "identifier",
    "ip",
    "date",
]

_CANONICAL_TO_TRAIN_LABEL: dict[str, str] = {
    "person": "person",
    "organization": "organization",
    "location": "location",
    "email": "email",
    "phone": "phone",
    "identifier": "identifier",
    "ip": "ip",
    "date": "date",
}


@dataclass(slots=True)
class TrainingDatasetBundle:
    dataset_name: str
    splits: list[str]
    samples: list[EvalSample]
    training_records: list[dict[str, Any]]
    labels: list[str]
    stats: dict[str, int]


@dataclass(slots=True)
class FinetuneRunResult:
    run_name: str
    base_model: str
    output_dir: str
    checkpoint_dir: str
    final_model_dir: str
    resolved_device: str
    precision_mode: str
    started_at_utc: str
    finished_at_utc: str
    epochs: float
    max_steps: int
    train_samples: int


@dataclass(slots=True)
class ModelEvalResult:
    threshold: float
    flat_ner: bool
    labels: list[str]
    aggregate: EvaluationAggregate
    sample_count: int
    elapsed_seconds: float

    @property
    def exact_canonical_f1(self) -> float:
        return self.aggregate.exact_canonical.f1

    @property
    def overlap_canonical_f1(self) -> float:
        return self.aggregate.overlap_canonical.f1


def discover_dataset_splits(dataset_name: str, cache_dir: str, hf_token: str | None) -> list[str]:
    try:
        from datasets import get_dataset_split_names
    except Exception as exc:
        raise RuntimeError("datasets package is required. Install with guardrails-service[eval].") from exc
    return list(get_dataset_split_names(dataset_name, token=hf_token, cache_dir=cache_dir))


def load_eval_samples_for_splits(
    dataset_name: str,
    splits: list[str],
    cache_dir: str,
    hf_token: str | None,
    max_samples_per_split: int | None = None,
) -> list[EvalSample]:
    adapter = get_dataset_adapter(dataset_name)
    all_samples: list[EvalSample] = []

    for split in splits:
        split_samples = adapter.load_samples(
            split=split,
            cache_dir=cache_dir,
            hf_token=hf_token,
            max_samples=max_samples_per_split,
        )
        for sample in split_samples:
            if "__split__" not in sample.metadata:
                sample.metadata["__split__"] = split
        all_samples.extend(split_samples)
    return all_samples


def _word_boundaries(
    text: str,
    splitter_type: str = "whitespace",
) -> tuple[list[str], list[int], list[int], dict[int, int], dict[int, int]]:
    try:
        from gliner.data_processing.tokenizer import WordsSplitter
    except Exception as exc:
        raise RuntimeError("gliner package is required. Install project dependencies (for example: `uv sync`).") from exc

    splitter = WordsSplitter(splitter_type=splitter_type)
    tokens: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    start_to_idx: dict[int, int] = {}
    end_to_idx: dict[int, int] = {}

    for idx, (token, start, end) in enumerate(splitter(text)):
        token_text = str(token)
        s = int(start)
        e = int(end)
        tokens.append(token_text)
        starts.append(s)
        ends.append(e)
        start_to_idx[s] = idx
        end_to_idx[e] = idx

    return tokens, starts, ends, start_to_idx, end_to_idx


def _char_to_word_span(
    start: int,
    end: int,
    starts: list[int],
    ends: list[int],
    start_to_idx: dict[int, int],
    end_to_idx: dict[int, int],
) -> tuple[int, int] | None:
    # Fast path: exact alignment to word boundaries.
    if start in start_to_idx and end in end_to_idx:
        ws = start_to_idx[start]
        we = end_to_idx[end]
        if we >= ws:
            return ws, we

    # Fallback path: use any overlapping token range.
    overlapping = [idx for idx, (s, e) in enumerate(zip(starts, ends, strict=False)) if s < end and e > start]
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1]


def build_training_bundle_from_eval_samples(
    dataset_name: str,
    splits: list[str],
    samples: list[EvalSample],
    splitter_type: str = "whitespace",
) -> TrainingDatasetBundle:
    records: list[dict[str, Any]] = []
    labels_set: set[str] = set()
    stats = {
        "total_samples": len(samples),
        "records_with_entities": 0,
        "records_without_entities": 0,
        "gold_spans_total": 0,
        "gold_spans_mapped": 0,
        "gold_spans_dropped_unmapped": 0,
        "gold_spans_dropped_unaligned": 0,
    }

    for sample in samples:
        tokens, starts, ends, start_to_idx, end_to_idx = _word_boundaries(sample.text, splitter_type=splitter_type)
        ner_set: set[tuple[int, int, str]] = set()

        for span in sample.gold_spans:
            stats["gold_spans_total"] += 1
            canonical = span.canonical_label
            if canonical is None:
                stats["gold_spans_dropped_unmapped"] += 1
                continue
            mapped_label = _CANONICAL_TO_TRAIN_LABEL.get(canonical)
            if mapped_label is None:
                stats["gold_spans_dropped_unmapped"] += 1
                continue

            word_span = _char_to_word_span(
                start=span.start,
                end=span.end,
                starts=starts,
                ends=ends,
                start_to_idx=start_to_idx,
                end_to_idx=end_to_idx,
            )
            if word_span is None:
                stats["gold_spans_dropped_unaligned"] += 1
                continue

            ws, we = word_span
            ner_set.add((ws, we, mapped_label))
            labels_set.add(mapped_label)
            stats["gold_spans_mapped"] += 1

        ner = sorted(ner_set, key=lambda item: (item[0], item[1], item[2]))
        record = {
            "sample_id": sample.sample_id,
            "tokenized_text": tokens,
            "ner": ner,
            "text": sample.text,
            "split": sample.metadata.get("__split__"),
        }
        records.append(record)
        if ner:
            stats["records_with_entities"] += 1
        else:
            stats["records_without_entities"] += 1

    labels = sorted(labels_set, key=lambda label: DEFAULT_TARGET_LABELS.index(label) if label in DEFAULT_TARGET_LABELS else 999)
    return TrainingDatasetBundle(
        dataset_name=dataset_name,
        splits=splits,
        samples=samples,
        training_records=records,
        labels=labels,
        stats=stats,
    )


def write_training_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> str:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)


def read_training_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _resolve_precision_mode(
    resolved_device: str,
    precision: str,
) -> tuple[bool, bool, str]:
    mode = precision.strip().lower()
    if mode not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("precision must be one of: auto, fp32, fp16, bf16")

    use_fp16 = False
    use_bf16 = False
    resolved_mode = "fp32"

    if mode == "fp32":
        return False, False, "fp32"
    if mode == "fp16":
        return True, False, "fp16"
    if mode == "bf16":
        return False, True, "bf16"

    if resolved_device == "cuda":
        try:
            import torch

            if bool(callable(getattr(torch.cuda, "is_bf16_supported", None)) and torch.cuda.is_bf16_supported()):
                use_bf16 = True
                resolved_mode = "bf16"
            else:
                use_fp16 = True
                resolved_mode = "fp16"
        except Exception:
            use_fp16 = True
            resolved_mode = "fp16"
    return use_fp16, use_bf16, resolved_mode


def finetune_gliner_records(
    *,
    records: list[dict[str, Any]],
    base_model: str,
    output_root: str,
    run_name: str,
    device: str = "auto",
    precision: str = "auto",
    num_train_epochs: float = 2.0,
    max_steps: int = -1,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.05,
    logging_steps: int = 25,
    save_total_limit: int = 2,
    dataloader_num_workers: int = 0,
    seed: int = 42,
    compile_model: bool = False,
    allow_cpu_fallback_on_oom: bool = True,
) -> FinetuneRunResult:
    try:
        from gliner import GLiNER
    except Exception as exc:
        raise RuntimeError("gliner package is required. Install project dependencies (for example: `uv sync`).") from exc

    if not records:
        raise ValueError("training records are empty")

    started = datetime.now(tz=UTC)
    resolved_device = resolve_torch_device(device)

    run_dir = Path(output_root).expanduser().resolve() / run_name
    ckpt_dir = run_dir / "checkpoints"
    final_dir = run_dir / "final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    device_attempts = [resolved_device]
    if allow_cpu_fallback_on_oom and resolved_device != "cpu":
        device_attempts.append("cpu")

    final_used_device = resolved_device
    final_precision = "fp32"
    last_exc: Exception | None = None

    for device_attempt in device_attempts:
        use_fp16, use_bf16, precision_mode = _resolve_precision_mode(device_attempt, precision)
        model = GLiNER.from_pretrained(base_model)
        if device_attempt in {"cuda", "mps"} and hasattr(model, "to"):
            model.to(device_attempt)

        training_args = model.create_training_args(
            output_dir=str(ckpt_dir),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=save_total_limit,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_cpu=device_attempt == "cpu",
            fp16=use_fp16,
            bf16=use_bf16,
            dataloader_num_workers=dataloader_num_workers,
            report_to="none",
            seed=seed,
        )
        try:
            model.train_model(
                train_dataset=records,
                eval_dataset=records,
                training_args=training_args,
                compile_model=compile_model,
            )
            model.save_pretrained(str(final_dir))
            final_used_device = device_attempt
            final_precision = precision_mode
            last_exc = None
            break
        except RuntimeError as exc:
            last_exc = exc
            message = str(exc).lower()
            is_oom = "out of memory" in message
            should_retry = is_oom and device_attempt != "cpu" and "cpu" in device_attempts
            if should_retry:
                continue
            raise

    if last_exc is not None:
        raise last_exc

    finished = datetime.now(tz=UTC)
    return FinetuneRunResult(
        run_name=run_name,
        base_model=base_model,
        output_dir=str(run_dir),
        checkpoint_dir=str(ckpt_dir),
        final_model_dir=str(final_dir),
        resolved_device=final_used_device,
        precision_mode=final_precision,
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        epochs=float(num_train_epochs),
        max_steps=int(max_steps),
        train_samples=len(records),
    )


def evaluate_model_on_eval_samples(
    *,
    model_ref: str,
    samples: list[EvalSample],
    labels: list[str],
    threshold: float = 0.5,
    flat_ner: bool = False,
    batch_size: int = 16,
    device: str = "auto",
    include_overlap: bool = True,
    include_per_label: bool = True,
) -> ModelEvalResult:
    try:
        from gliner import GLiNER
    except Exception as exc:
        raise RuntimeError("gliner package is required. Install project dependencies (for example: `uv sync`).") from exc

    if not samples:
        raise ValueError("evaluation samples are empty")

    model = GLiNER.from_pretrained(model_ref)
    resolved_device = resolve_torch_device(device)
    if resolved_device in {"cuda", "mps"} and hasattr(model, "to"):
        model.to(resolved_device)

    started = datetime.now(tz=UTC)
    predictions_by_id: dict[str, list[EvalSpan]] = {}

    for offset in range(0, len(samples), batch_size):
        batch_samples = samples[offset : offset + batch_size]
        batch_texts = [sample.text for sample in batch_samples]
        raw_preds = model.inference(
            batch_texts,
            labels=labels,
            threshold=threshold,
            flat_ner=flat_ner,
            batch_size=batch_size,
        )
        for sample, sample_preds in zip(batch_samples, raw_preds, strict=False):
            predicted_spans = [
                EvalSpan(
                    start=int(item["start"]),
                    end=int(item["end"]),
                    label=str(item["label"]),
                    canonical_label=canonicalize_prediction_label(str(item["label"])),
                    score=float(item.get("score", 0.0)),
                    detector="gliner_finetuned",
                )
                for item in sample_preds
            ]
            predictions_by_id[sample.sample_id] = predicted_spans

    aggregate = evaluate_samples(
        samples,
        predictions_by_id,
        include_overlap=include_overlap,
        include_per_label=include_per_label,
    )
    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    return ModelEvalResult(
        threshold=threshold,
        flat_ner=flat_ner,
        labels=labels,
        aggregate=aggregate,
        sample_count=len(samples),
        elapsed_seconds=elapsed,
    )


def evaluate_model_builtin_f1(
    *,
    model_ref: str,
    records: list[dict[str, Any]],
    threshold: float = 0.5,
    flat_ner: bool = False,
    batch_size: int = 16,
    device: str = "auto",
) -> tuple[float, dict[str, Any]]:
    try:
        from gliner import GLiNER
    except Exception as exc:
        raise RuntimeError("gliner package is required. Install project dependencies (for example: `uv sync`).") from exc

    model = GLiNER.from_pretrained(model_ref)
    resolved_device = resolve_torch_device(device)
    if resolved_device in {"cuda", "mps"} and hasattr(model, "to"):
        model.to(resolved_device)

    output, f1 = model.evaluate(
        test_data=records,
        flat_ner=flat_ner,
        threshold=threshold,
        batch_size=batch_size,
    )
    if isinstance(output, dict):
        return float(f1), output
    return float(f1), {"raw_output": output}
