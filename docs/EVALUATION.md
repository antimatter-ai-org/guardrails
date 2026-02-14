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
- `per_label_exact`: exact canonical metrics by label.

Canonical labels are normalized from:
- dataset labels (adapter mapping)
- guardrails prediction labels (`app/eval/labels.py`)

## Report Format

Each run emits:

- JSON: machine-readable report
- Markdown: human summary

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

## Caching and Auth

- Dataset cache default: `.eval_cache/hf`
- Env file default: `.env.eval`
- Token variable: `HF_TOKEN`

The env file is loaded automatically by the CLI (`--env-file` option).
