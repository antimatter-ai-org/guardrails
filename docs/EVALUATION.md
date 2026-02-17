# Evaluation System (v3)

This project ships a suite-based, task-based evaluation system under `app/eval_v3`.

Goals:
- Evaluate a **suite** of datasets by default (not a hardcoded shortlist).
- Score datasets **only on labels they supervise** (missing labels do not skew combined scores).
- Produce comprehensive reports (JSON source of truth + human Markdown + machine-readable CSV).
- Support CPU/MPS/CUDA runtime modes (via existing runtime envs).
- Support offline/air-gapped operation with explicit dataset pre-download.

## Quick Start

Setup:

```bash
uv sync --extra dev --extra eval
cp .env.eval.example .env.eval
# set HF_TOKEN=... in .env.eval (required because suite contains private datasets)
```

Run the default suite on `fast`:

```bash
uv run --extra eval python -m app.eval_v3.cli \
  --suite guardrails_ru \
  --split fast \
  --policy-path configs/policy.yaml \
  --policy-name external_default \
  --env-file .env.eval \
  --output-dir reports/evaluations
```

Run on `full`:

```bash
uv run --extra eval python -m app.eval_v3.cli --suite guardrails_ru --split full --env-file .env.eval
```

Run a subset of datasets:

```bash
uv run --extra eval python -m app.eval_v3.cli \
  --dataset antimatter-ai/guardrails-ru-scanpatch-pii-ner-controlled-v1 \
  --dataset antimatter-ai/guardrails-ru-presidio-test-dataset-v1 \
  --split fast --env-file .env.eval
```

## Suite Registry

Suites and dataset configs live in:
- `configs/eval/suites.yaml`

Adding a dataset is done by adding a new entry in that YAML:
- `datasets.<dataset_id>.scored_labels` describes which labels are supervised and should be included in scoring.
- `datasets.<dataset_id>.label_map` maps dataset-specific gold labels to canonical labels.

## Tasks

The eval runner is task-based. Default `--tasks all` runs:

1. `span_detection`
- Span-level NER metrics on canonical labels.
- Metrics families:
  - `overlap_canonical` (primary “miss/leakage” KPI for entity coverage)
  - `char_canonical` (coverage proxy: how much sensitive text is covered)
  - `token_canonical` (coverage proxy at token granularity)
  - `exact_canonical` (strict boundary quality indicator)

2. `policy_action`
- Binary classification: should the policy intervene (`NONE` vs `MASKED/BLOCKED`) given label-supervised gold.
- Evaluated per policy id (default `external_default,strict_block`), using each policy’s analyzer profile.

3. `mask_leakage` (diagnostic)
- Runs deterministic in-process masking using predicted spans.
- Reports approximate “gold span leaked verbatim” fraction and sample previews.

## Label-Aware Scoring (Critical)

Each dataset has a configured `scored_labels` set.

For scoring:
- Gold spans are filtered to `scored_labels`.
- Predicted spans are filtered to `scored_labels`.
- Predicted spans outside `scored_labels` are **ignored for FP/FN** and reported separately under `unscored_predictions`.

This prevents datasets with incomplete supervision from skewing combined suite scores.

## Runtime Modes (CPU/MPS/CUDA)

Eval v3 uses the same runtime stack as production guardrails.

Select runtime via env or CLI:
- `GR_RUNTIME_MODE=cpu` (default)
- `GR_RUNTIME_MODE=cuda`

On Apple Silicon CPU mode, device is controlled by:
- `GR_CPU_DEVICE=auto` (prefers MPS if available)
- `GR_CPU_DEVICE=mps`
- `GR_CPU_DEVICE=cpu`

## Offline / Air-Gapped Datasets

Pre-download suite datasets into a local directory:

```bash
make download-datasets DATASETS_DIR=./.datasets
```

This creates:
- `./.datasets/hf_cache/...` (HF hub + datasets cache)
- `./.datasets/manifest.json`

Then run offline by pointing HF cache env vars to that cache and enabling offline mode:

```bash
export HF_HOME=./.datasets/hf_cache
export HUGGINGFACE_HUB_CACHE=./.datasets/hf_cache/hub
export HF_DATASETS_CACHE=./.datasets/hf_cache/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

uv run --extra eval python -m app.eval_v3.cli --suite guardrails_ru --split fast --offline --env-file .env.eval
```

## Report Outputs

Each run creates a directory:
- `reports/evaluations/<run_id>/`

Files:
- `report.json` (source of truth)
- `report.md` (human summary)
- `metrics.csv` (machine-readable summary rows)

## Comparing Reports

Use the compare tool:

```bash
uv run --extra eval python -m app.eval_v3.compare \
  --base reports/evaluations/<old_run>/report.json \
  --new  reports/evaluations/<new_run>/report.json \
  --out  reports/evaluations/diff_<name>
```

## Legacy Evaluator

The previous evaluator remains at `app/eval/run.py` for now, but `make eval-all` uses v3 by default.

