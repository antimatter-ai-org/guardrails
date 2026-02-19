# Evaluation

## Framework

Canonical eval entrypoints:
- `python -m app.eval.cli`
- `python -m app.eval.compare`
- `python -m app.tools.merge_eval_reports`

Suite/dataset registry:
- `configs/eval/suites.yaml`

## Environment

Install deps:

```bash
uv sync --extra eval
```

Optional env file:

```bash
cp .env.eval.example .env.eval
```

## Run Eval

Full RU suite (fast split):

```bash
uv run --extra eval python -m app.eval.cli \
  --suite guardrails_ru \
  --split fast \
  --policy-path configs/policy.yaml \
  --policy-name external \
  --action-policies external \
  --env-file .env.eval \
  --output-dir reports/evaluations
```

Single dataset override:

```bash
uv run --extra eval python -m app.eval.cli \
  --suite guardrails_ru \
  --dataset antimatter-ai/guardrails-ru-scanpatch-pii-ner-controlled-v1 \
  --split fast \
  --policy-name external \
  --action-policies external \
  --env-file .env.eval \
  --output-dir reports/evaluations
```

CUDA run:

```bash
GR_RUNTIME_MODE=cuda uv run --extra eval --extra cuda python -m app.eval.cli --suite guardrails_ru --split fast --policy-name external --action-policies external --env-file .env.eval
```

## Outputs

Each run creates `reports/evaluations/<run_id>/`:
- `report.json`
- `report.md`
- `metrics.csv`

## Compare Runs

```bash
uv run --extra eval python -m app.eval.compare \
  --base /path/to/base/report.json \
  --new /path/to/new/report.json \
  --out /path/to/diff
```

Outputs:
- `/path/to/diff/diff.json`
- `/path/to/diff/diff.md`

## Merge Split Runs

```bash
uv run --extra eval python -m app.tools.merge_eval_reports \
  --out /path/to/merged.json \
  /path/to/report_gpu0.json \
  /path/to/report_gpu1.json
```

Validation rules enforced by merger:
- same `report_version`
- same `suite`
- same `split`
- same `policy_name`

## Baseline Retention

One pre-cleanup baseline report is retained under `docs/evals/runs/` for post-cleanup comparison.
