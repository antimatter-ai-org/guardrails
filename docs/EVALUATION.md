# Evaluation System

Manual evaluation framework for guardrails detection quality on public datasets.

## Scope

- Single evaluator entrypoint: `python -m app.eval.run`
- Runs one dataset (`--dataset`) or all supported datasets (default)
- Uses the same analysis service and policy logic as runtime guardrails
- Produces unified JSON + Markdown reports
- `script_profile` is an evaluation-only slice for reporting; runtime guardrails do not use script/language routing.

## Supported Dataset Behavior

- Automatic dataset download from Hugging Face.
- Local cache reused across runs.
- For datasets without native `test`, evaluator requests synthetic `test` from adapter:
  - split generated from full `train`
  - label-balance heuristic applied
  - generated indices cached and reused

## CLI

Arguments:

- `--dataset`: dataset name; repeatable. If omitted, evaluator runs all supported datasets.
- `--split`: requested split name (default `test`).
- `--policy-path`: path to policy YAML (default `configs/policy.yaml`).
- `--policy-name`: policy id from YAML. If omitted, evaluator uses `default_policy`.
- `--cache-dir`: Hugging Face/datasets cache directory (default `.eval_cache/hf`).
- `--output-dir`: directory for JSON/Markdown reports (default `reports/evaluations`).
- `--env-file`: env file loaded before run (default `.env.eval`).
- `--hf-token-env`: env var name holding HF token (default `HF_TOKEN`).
- `--strict-split`: do not fall back to unrelated native split when requested split is missing.
- `--no-strict-split`: allow fallback to another available native split (typically `train`).
- `--synthetic-test-size`: test ratio for generated synthetic split on datasets without native test (default `0.2`).
- `--synthetic-split-seed`: random seed for synthetic split generation/caching (default `42`).
- `--max-samples`: optional cap on evaluated samples per dataset.
- `--errors-preview-limit`: max number of mismatch examples saved in report (default `25`).
- `--progress-every-samples`: print progress every N processed samples (default `1000`).
- `--progress-every-seconds`: also print progress every N seconds (default `15.0`).

## Report Outputs

Each run writes:

- JSON report
- Markdown summary

The JSON report is the source of truth. Markdown is a human summary derived from JSON.

## Built-In Regression Gates

Evaluator enforces a default quality gate for full test-split runs on:
- `BoburAmirov/rubai-NER-150K-Personal`

Gate thresholds:
- `per_label_exact.person.precision >= 0.30`
- `per_label_exact.person.f1 >= 0.35`

Behavior:
- Gate is evaluated only for full runs (when `--max-samples` is not set).
- If gate fails, evaluator exits with a non-zero error after writing reports.
- This is intended to catch severe person-label regressions early.

## JSON Schema Reference

Top-level payload:

- `report_version`: schema version of the report format.
- `generated_at_utc`: UTC timestamp when the report payload was generated.
- `dataset`: summary of the evaluated dataset scope.
- `evaluation`: runtime/policy metadata and throughput data.
- `metrics`: combined metrics for the full evaluated set.
- `errors_preview`: sample-level mismatch preview.
- `datasets`: per-dataset report sections (present when one or more datasets were evaluated).
- `dataset_slices`: combined slice-level metrics.
- `detector_breakdown`: combined detector-level contribution metrics.

`dataset` payload:

- `name`: dataset name, or `all` for multi-dataset runs.
- `split`: requested split name.
- `sample_count`: number of evaluated samples in this combined section.

`evaluation` payload:

- `policy_name`: policy used for detection.
- `policy_path`: policy file path.
- `runtime_mode`: `cpu` or `cuda`.
- `elapsed_seconds`: wall-clock evaluation duration.
- `samples_per_second`: throughput (samples divided by elapsed time).
- `mode`: evaluator mode (`baseline`).
- `generated_at_utc`: UTC timestamp for evaluation block generation.

`datasets[]` item payload (per dataset):

- `name`: dataset id.
- `split`: actual split used after split resolution.
- `available_splits`: splits reported by dataset loader.
- `sample_count`: evaluated sample count for this dataset.
- `elapsed_seconds`: dataset-specific wall-clock duration.
- `samples_per_second`: dataset-specific throughput.
- `metrics`: metric payload for this dataset.
- `errors_preview`: dataset-specific mismatch preview.
- `dataset_slices`: dataset-specific slice metrics.
- `detector_breakdown`: dataset-specific detector contribution metrics.

`errors_preview[]` item payload:

- `sample_id`: dataset sample id.
- `false_positives`: number of predicted spans not matched to gold.
- `false_negatives`: number of gold spans missed by prediction.
- `text`: original sample text.

`dataset_slices` payload:

- `source`: metrics grouped by `sample.metadata["source"]`.
- `noisy`: metrics grouped by noise marker (`true`/`false`/`unknown`).
- `script_profile`: metrics grouped by script mix:
  - `mostly_cyrillic`
  - `mostly_latin`
  - `mixed`
  - `no_letters`

Each group item contains:

- `sample_count`
- `exact_canonical`
- `overlap_canonical`

`detector_breakdown` payload:

- keys: detector/recognizer names from prediction metadata.
- values:
  - `prediction_count`: raw count of spans emitted by detector.
  - `canonical_prediction_count`: spans with canonical label.
  - `overlap_agnostic`: overlap metric payload ignoring labels.
  - `overlap_canonical`: overlap metric payload with canonical labels.

## Metrics

Related report components outside `metrics`:

- `datasets` (per-dataset metrics and metadata)
- `detector_breakdown`
- `dataset_slices`
- `errors_preview`

Metric families and meanings:

- `exact_agnostic`: exact boundary match, labels ignored.
- `overlap_agnostic`: span overlap match, labels ignored.
- `exact_canonical`: exact boundary + canonical label match.
- `overlap_canonical`: overlap + canonical label match.
- `char_canonical`: character-level overlap on canonical labeled spans.
- `token_canonical`: token-level overlap on canonical labeled spans.
- `per_label_exact`: exact canonical metrics broken down by label.
- `per_label_char`: character-level canonical metrics broken down by label.

Fields inside every metric payload:

- `true_positives` (`TP`): predictions matched to gold spans.
- `false_positives` (`FP`): predictions not matched to gold spans.
- `false_negatives` (`FN`): gold spans missed by predictions.
- `precision`: `TP / (TP + FP)`; higher means fewer false alarms.
- `recall`: `TP / (TP + FN)`; higher means fewer misses.
- `f1`: harmonic mean of precision and recall.
- `residual_miss_ratio`: `1 - recall`; direct miss/leakage proxy.

Practical interpretation:

- Exact metrics are strict boundary quality indicators.
- Overlap metrics are robust to minor boundary drift.
- Character and token metrics are leakage-centric proxies for partial masking coverage.

## Metrics to Prioritize for Real Router Performance

For guardrails in an LLM router, missed sensitive spans are usually more costly than extra masking.  
Use the following priority order:

1. `overlap_canonical.recall` and `overlap_canonical.residual_miss_ratio`
- Best first-line indicator of missed sensitive entities when minor boundary shifts are acceptable.
- Primary leakage-risk KPI.

2. `char_canonical.recall` and `char_canonical.residual_miss_ratio`
- Measures how much sensitive text is actually covered.
- Important when partial span misses still leak meaningful substrings.

3. `per_label_char` for highest-risk labels
- Track labels like `secret`, `payment_card`, `identifier`, `email`, `phone` separately.
- Catch regressions hidden by good combined metrics.

4. `token_canonical.recall`
- Useful for long or multi-token entities where exact boundaries are hard.
- Good secondary leakage proxy.

5. `precision` (especially canonical overlap/char)
- Controls over-masking and user experience.
- Tune after recall safety floor is met.

6. `samples_per_second`
- Operational capacity metric.
- Use together with quality metrics; do not trade away recall blindly.

Suggested operational workflow:

1. Gate releases on recall-oriented metrics (`overlap_canonical`, `char_canonical`, high-risk `per_label_char`).
2. Monitor precision and detector breakdown to reduce noise.
3. Use `dataset_slices` to ensure stable behavior across Cyrillic/Latin/mixed traffic.
4. Investigate `errors_preview` for recurring false-negative patterns and update recognizers/patterns.

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
