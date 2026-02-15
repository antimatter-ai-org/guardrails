# Evaluation System

This repository includes a manual evaluation framework to benchmark guardrails recognizers on public datasets.

## Goals

- Run with one command.
- Auto-download datasets.
- Keep datasets in local cache.
- Produce consistent report format for future dataset adapters.
- No CI integration required.

## Entry Point

- CLI: `python -m app.eval.run`
- Shortcut: `make eval-scanpatch`
- Baseline shortcut: `make eval-scanpatch-baseline`
- Cascade shortcut: `make eval-scanpatch-cascade`
- Baseline manifest: `make eval-manifest EVAL_BASE_REPORT=<report.json>`
- Report comparison: `make eval-compare EVAL_BASE_REPORT=<base.json> EVAL_CANDIDATE_REPORT=<candidate.json>`
- Policy matrix runner: `make eval-matrix EVAL_POLICY_ARGS="--policy-name external_default --policy-name strict_block"`

Dataset selection:
- `--dataset` omitted: run on all supported datasets.
- `--dataset <name>`: run on a specific dataset (repeatable option).
- If requested split is missing and adapter supports it, evaluator can build cached synthetic train/test split.
- Use `--strict-split` to avoid fallback to unrelated splits; synthetic split still works for supported adapters.
- Progress is printed during execution (`[progress] ...`) with processed count, throughput, and ETA.
- Progress cadence is configurable via `--progress-every-samples` and `--progress-every-seconds`.
- Warm-up behavior for runtime-backed recognizers is configurable:
  - `--warmup-timeout-seconds`
  - `--warmup-strict / --no-warmup-strict`

Default evaluator mode is `baseline`:
- All configured recognizers from the selected analyzer profile run on every sample.

Optional `cascade` mode:
- Stage A runs lightweight recognizers.
- Stage B runs heavy recognizers (GLiNER by default) only for uncertain samples.
- Use `--mode cascade` to enable staged execution.

## Adapter Design

Dataset integration is adapter-based:

- `app/eval/datasets/base.py`: adapter contract.
- `app/eval/datasets/registry.py`: dataset-name -> adapter mapping.
- `app/eval/datasets/scanpatch.py`: first adapter implementation.

To add a new dataset:
1. Implement a new adapter returning `EvalSample` objects.
2. Register it in `app/eval/datasets/registry.py`.
3. (Optional) add canonical label mapping in `app/eval/labels.py`.

## Metrics

Reports include:

- `exact_agnostic`: exact span match, labels ignored.
- `overlap_agnostic`: overlap span match, labels ignored.
- `exact_canonical`: exact span + canonical label match.
- `overlap_canonical`: overlap span + canonical label match.
- `char_canonical`: character-level overlap metrics on canonical spans.
- `token_canonical`: token-level overlap metrics on canonical spans.
- `per_label_exact`: exact canonical metrics by label.
- `per_label_char`: character-level canonical metrics by label.

Each metric payload includes:
- `precision`
- `recall`
- `f1`
- `residual_miss_ratio` (computed as `1 - recall`)

Canonical labels are normalized from:
- dataset labels (adapter mapping)
- guardrails prediction labels (`app/eval/labels.py`)

## Report Format

Each run emits:

- JSON: machine-readable report
- Markdown: human summary
- includes both combined metrics and per-dataset metric sections when multiple datasets are evaluated.

Top-level JSON fields:
- `report_version`
- `generated_at_utc`
- `dataset`
- `evaluation`
- `metrics`
- `errors_preview`
- `detector_breakdown` (grouped by recognizer/detector name in prediction metadata)
- `dataset_slices`

`evaluation` additions:
- `mode`: `baseline` or `cascade`
- `cascade` (only for cascade mode): threshold, stage profiles, heavy recognizers, escalation count/ratio.
- `warmup`: warm-up timeout/strict settings and per-recognizer readiness diagnostics.

## Baseline Manifest and Diff Workflow

To lock a baseline for future experiments:

```bash
make eval-manifest EVAL_BASE_REPORT=reports/evaluations/eval_all_datasets_test_baseline_YYYYMMDDTHHMMSSZ.json
```

This writes `reports/evaluations/baseline_manifest.json` with:
- report paths
- dataset/split metadata
- policy/runtime metadata
- current git commit SHA

To compare a candidate run against the baseline:

```bash
make eval-compare \
  EVAL_BASE_REPORT=reports/evaluations/eval_all_datasets_test_baseline_YYYYMMDDTHHMMSSZ.json \
  EVAL_CANDIDATE_REPORT=reports/evaluations/eval_all_datasets_test_baseline_YYYYMMDDTHHMMSSZ.json \
  EVAL_COMPARISON_OUTPUT=reports/evaluations/comparison_latest.md
```

If `EVAL_COMPARISON_OUTPUT` is omitted, markdown is printed to stdout.

To run multiple policies in one command and auto-compare candidates against the first report:

```bash
make eval-matrix \
  EVAL_POLICY_ARGS="--policy-name external_default --policy-name strict_block" \
  EVAL_COMPARISON_OUTPUT=reports/evaluations/comparison_matrix.md
```

To include recognizer ablations and resume support:

```bash
make eval-matrix \
  EVAL_POLICY_ARGS="--policy-name external_default" \
  EVAL_ABLATION_ARGS="--ablate-recognizer gliner_pii_multilingual --ablate-recognizer identifier_regex" \
  EVAL_RESUME=--resume \
  EVAL_COMPARISON_OUTPUT=reports/evaluations/comparison_matrix_ablation.md
```

`app.eval.run` checkpoint/resume options:
- `--resume`: load dataset-level checkpoint and skip completed datasets.
- `--checkpoint-path`: optional explicit checkpoint JSON path.
- Default checkpoint location: `reports/evaluations/_checkpoints/eval_checkpoint_<hash>.json`.

## Caching and Auth

- Dataset cache default: `.eval_cache/hf`
- Synthetic split cache: `.eval_cache/hf/_synthetic_splits/`
- Env file default: `.env.eval`
- Token variable: `HF_TOKEN`

The env file is loaded automatically by the CLI (`--env-file` option).
