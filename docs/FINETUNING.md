# GLiNER Fine-tuning

This repository includes manual scripts for GLiNER fine-tuning experiments.

Important:
- This MVP pipeline is intentionally configured for feasibility checks.
- It can train and evaluate on the same data (train-on-all/eval-on-all).
- Do not treat these metrics as generalization performance.

## Dependencies

Install with fine-tuning extras:

```bash
pip install -e '.[ml,eval,finetune,dev]'
```

## 1) Prepare training data from Scanpatch

This converts Scanpatch spans into GLiNER training format (`tokenized_text` + `ner`).
All available splits are used.

```bash
python -m app.tools.prepare_gliner_scanpatch_data \
  --dataset scanpatch/pii-ner-corpus-synthetic-controlled \
  --env-file .env.eval
```

Output defaults to:
- `reports/finetune/scanpatch_all_splits_gliner_train.jsonl`
- `reports/finetune/scanpatch_all_splits_gliner_train_summary.json`

## 2) Fine-tune model manually

```bash
python -m app.tools.finetune_gliner \
  --train-jsonl reports/finetune/scanpatch_all_splits_gliner_train.jsonl \
  --base-model urchade/gliner_multi-v2.1 \
  --run-name scanpatch_try_01 \
  --output-dir reports/finetune/runs \
  --device auto \
  --precision auto \
  --num-train-epochs 2.0
```

## 3) Run end-to-end pipeline (prepare + iterative fine-tune + eval)

```bash
python -m app.tools.run_scanpatch_gliner_finetune_pipeline \
  --dataset scanpatch/pii-ner-corpus-synthetic-controlled \
  --env-file .env.eval \
  --output-dir reports/finetune/scanpatch_pipeline \
  --iterations 2 \
  --epoch-schedule 1.0,2.0 \
  --thresholds 0.25,0.35,0.5 \
  --flat-ner
```

For large train-on-all/eval-on-all runs, use fast exact metrics to avoid very slow overlap/per-label aggregation:

```bash
python -m app.tools.run_scanpatch_gliner_finetune_pipeline \
  --dataset scanpatch/pii-ner-corpus-synthetic-controlled \
  --env-file .env.eval \
  --output-dir reports/finetune/scanpatch_pipeline \
  --iterations 1 \
  --thresholds 0.5 \
  --flat-ner \
  --eval-mode guardrails \
  --skip-overlap-metrics \
  --skip-per-label-metrics
```

This produces:
- per-iteration training artifacts under `reports/finetune/scanpatch_pipeline/runs/`
- final report JSON + Markdown under `reports/finetune/scanpatch_pipeline/`

## 4) Evaluate an existing checkpoint (no training)

```bash
python -m app.tools.evaluate_finetuned_gliner \
  --model-ref reports/finetune/scanpatch_pipeline/runs/iter_01/final \
  --dataset scanpatch/pii-ner-corpus-synthetic-controlled \
  --env-file .env.eval \
  --output-dir reports/finetune/eval \
  --flat-ner \
  --skip-overlap-metrics \
  --skip-per-label-metrics
```

This writes JSON + Markdown reports under `reports/finetune/eval/`.

## GPU notes

- `--device auto` picks `cuda` when available, then `mps`, otherwise `cpu`.
- To force GPU: `--device cuda`.
- Precision:
  - `--precision auto` picks `bf16` on supported CUDA devices, otherwise `fp16` on CUDA, otherwise `fp32`.
  - You can force `fp32|fp16|bf16`.
- For larger runs on GPU, tune:
  - `--per-device-train-batch-size`
  - `--gradient-accumulation-steps`
  - `--dataloader-num-workers`
