# Evaluation System (v3)

This project includes a suite-driven evaluation harness that measures Guardrails detection quality on Hugging Face datasets.

## Goals

- Use **multiple datasets** without skewing suite scores when a dataset does not annotate some labels.
- Provide **repeatable** runs via cached dataset downloads and cached derived views/splits.
- Produce both **human** and **machine-readable** reports suitable for regression tracking and comparisons.

## Key Concepts

### Canonical Labels

Canonical PII labels are defined in:
- `/Users/oleg/Projects/_antimatter/guardrails/app/core/labels.py`

Runtime canonicalization is handled by:
- `/Users/oleg/Projects/_antimatter/guardrails/app/core/analysis/mapping.py`

### Suites

A suite is a YAML file that declares:
- which datasets belong to the suite
- which labels are in-scope for scoring
- for each dataset: which canonical labels are actually annotated, and how to map gold labels to canonical labels

Default suite:
- `/Users/oleg/Projects/_antimatter/guardrails/configs/eval/suites/guardrails-ru.yaml`

### Label-Safe Scoring (Prevents Partial-Annotation Skew)

Each dataset has an explicit `annotated_labels` list.

During scoring for a dataset:
1. Gold spans are kept only if their canonical label is in `annotated_labels ∩ scored_labels`.
2. Predicted spans are kept only if their canonical label is in `annotated_labels ∩ scored_labels`.

This prevents a common failure mode: if a dataset does not annotate (say) `secret`, and your model predicts `secret`, we do **not** count that as a false positive for that dataset.

At suite-level:
- Per-label counts are aggregated only from datasets that annotate that label.
- The headline KPI uses only labels with **non-zero gold support** (otherwise a label would be “N/A”, not “0 recall”).

### Views (Cached Derived Splits)

Views are cached subsets of a dataset split (filtering and/or sampling). They are stored under:
- `.eval_cache/views/<dataset_slug>/<view_id>.json`

Views are re-used across runs when the dataset fingerprint matches.

Built-in view:
- `negative`: rows where `__scored_entity_count__ == 0` (good for false-positive pressure testing)

## CLI

Entrypoint:

```bash
uv run --extra eval python -m app.eval.run --help
```

### Default Run (All Datasets, fast split)

```bash
uv run --extra eval python -m app.eval.run \
  --suite guardrails-ru \
  --env-file .env.eval \
  --output-dir reports/evaluations
```

### Full Run (All Datasets, full split)

```bash
uv run --extra eval python -m app.eval.run \
  --suite guardrails-ru \
  --split full \
  --env-file .env.eval
```

### One Dataset

```bash
uv run --extra eval python -m app.eval.run \
  --suite guardrails-ru \
  --dataset antimatter-ai/guardrails-ru-presidio-test-dataset-v1 \
  --split fast \
  --env-file .env.eval
```

### Tag Filter (AND semantics)

```bash
uv run --extra eval python -m app.eval.run \
  --suite guardrails-ru \
  --tag en --tag secrets \
  --split fast \
  --env-file .env.eval
```

### Cached View: Negative Slice

```bash
uv run --extra eval python -m app.eval.run \
  --suite guardrails-ru \
  --view negative \
  --split fast \
  --env-file .env.eval
```

### Custom Filter + Sampling (Cached)

```bash
uv run --extra eval python -m app.eval.run \
  --suite guardrails-ru \
  --where language=en \
  --max-samples 2000 \
  --seed 123 \
  --stratify-by language,script_profile,label_presence \
  --split fast \
  --env-file .env.eval
```

### Comparison Run

```bash
uv run --extra eval python -m app.eval.run \
  --suite guardrails-ru \
  --split fast \
  --compare /path/to/old/run/report.json \
  --env-file .env.eval
```

## CPU / MPS / CUDA

- CPU/MPS (Apple Silicon):
  - `--runtime-mode cpu`
  - `--cpu-device auto|mps|cpu` (default behavior is `auto`)
- CUDA:
  - `--runtime-mode cuda`

These flags set `GR_RUNTIME_MODE` / `GR_CPU_DEVICE` for the evaluation process, and the evaluator uses the same model runtime wiring as production Guardrails.

## Outputs

Each run writes to:
- `reports/evaluations/<run_id>/report.json` (source of truth)
- `reports/evaluations/<run_id>/summary.md` (human summary)
- `reports/evaluations/<run_id>/metrics.csv` (flat table)
- `reports/evaluations/<run_id>/errors.jsonl` (all mismatch samples with gold + predicted spans)
- `reports/evaluations/<run_id>/config.resolved.json` (resolved suite + args + cache paths)
- `reports/evaluations/<run_id>/comparison.json` (only when `--compare` is used)

## Metrics (What to Look At)

Guardrails is a safety component: **misses are more costly than extra masking**.

Recommended priority:
1. `char_canonical.recall` (leakage proxy): how much sensitive text was covered.
2. `overlap_canonical.recall`: robust to minor boundary drift.
3. Per-label `char_canonical.recall` for high-risk labels (e.g. `secret`, `payment_card`, `identifier`, `email`, `phone`).
4. Precision: controls over-masking and UX impact once recall floor is acceptable.

The suite headline KPI is:
- **risk-weighted macro** of per-label `char_canonical.recall`, skipping labels with no gold support.

Weights live in:
- `/Users/oleg/Projects/_antimatter/guardrails/configs/eval/weights.yaml`

## Gates (Optional)

Gates are optional thresholds for regression control:
- `/Users/oleg/Projects/_antimatter/guardrails/configs/eval/gates.yaml`

By default gates are enforced only on full, non-sampled “whole suite” runs, unless `--enforce-gates` is set.

## Adding a Dataset

1. Add the dataset to the suite YAML:
   - `/Users/oleg/Projects/_antimatter/guardrails/configs/eval/suites/guardrails-ru.yaml`
2. Provide:
   - `annotated_labels`: canonical labels the dataset truly annotates
   - `gold_label_mapping`: raw gold label -> canonical label mapping
   - `slice_fields`: fields you want available for slicing/reporting
   - `tags` and `notes` for selection and reporting

If the dataset follows the standard schema (`source_text`, `privacy_mask`, `fast`/`full` parquet splits), no code changes are required.

