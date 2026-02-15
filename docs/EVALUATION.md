# Evaluation System

Manual evaluation framework for guardrails detection quality on public datasets.

## Scope

- Single evaluator entrypoint: `python -m app.eval.run`
- Runs one dataset (`--dataset`) or all supported datasets (default)
- Uses the same analysis service and policy logic as runtime guardrails
- Produces unified JSON + Markdown reports

## Supported Dataset Behavior

- Automatic dataset download from Hugging Face.
- Local cache reused across runs.
- For datasets without native `test`, evaluator requests synthetic `test` from adapter:
  - split generated from full `train`
  - label-balance heuristic applied
  - generated indices cached and reused

## CLI

Primary arguments:

- `--dataset` repeatable dataset name
- `--split` (default `test`)
- `--policy-path`
- `--policy-name`
- `--cache-dir`
- `--output-dir`
- `--env-file`
- `--hf-token-env`
- `--strict-split / --no-strict-split`
- `--synthetic-test-size`
- `--synthetic-split-seed`
- `--max-samples`
- `--progress-every-samples`
- `--progress-every-seconds`

## Reports

Each run writes:

- JSON report
- Markdown summary

Report includes:

- combined metrics
- per-dataset metrics
- per-label metrics
- detector breakdown
- dataset slices (`source`, `noisy`, `script_profile`)
- errors preview

Metric families:

- `exact_agnostic`
- `overlap_agnostic`
- `exact_canonical`
- `overlap_canonical`
- `char_canonical`
- `token_canonical`
- `per_label_exact`
- `per_label_char`

Each metric payload includes:

- `precision`
- `recall`
- `f1`
- `residual_miss_ratio`
- `true_positives`
- `false_positives`
- `false_negatives`

## Usage Examples

Run all datasets:

```bash
uv run --extra eval python -m app.eval.run --split test --policy-path configs/policy.yaml --policy-name external_default --env-file .env.eval --output-dir reports/evaluations
```

Run one dataset:

```bash
uv run --extra eval python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test --env-file .env.eval
```

Strict split behavior:

```bash
uv run --extra eval python -m app.eval.run --dataset BoburAmirov/rubai-NER-150K-Personal --split test --strict-split
```

Progress output is emitted during run:

```text
[progress] dataset=... processed=... rate=... elapsed=... eta=...
```
