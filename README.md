# Guardrails Service (MVP)

Guardrails microservice for LLM routers.

This service does detection, masking, and unmasking only. It does not route LLM traffic.

## Architecture

Router call sequence:
1. Call `mask` before sending content to an external LLM.
2. Send masked content through your router/upstream.
3. Call `unmask` (non-streaming) or `unmask-stream` (streaming).
4. Call `finalize` on completion/cancel/error.

## Features

- Reversible masking with Redis-backed request state.
- RU/EN recognizer stack with Presidio backend, including GLiNER enabled by default.
- Streaming-safe unmasking with chunk boundary buffering.
- Global runtime switch:
  - `cpu`: in-process model execution (tries MPS on Apple Silicon)
  - `cuda`: model execution through PyTriton

## API

- `GET /v1/guardrails/policies`
- `POST /v1/guardrails/detect`
- `POST /v1/guardrails/mask`
- `POST /v1/guardrails/unmask`
- `POST /v1/guardrails/unmask-stream`
- `POST /v1/guardrails/finalize`

## Runtime modes

### CPU mode

- `GR_RUNTIME_MODE=cpu`
- GLiNER runs in-process with torch.
- Local torch device preference is controlled by `GR_CPU_DEVICE` (`auto` by default; prefers `mps` on Apple Silicon).
- GLiNER and Natasha can be loaded from local model directory via `GR_MODEL_DIR`.

### CUDA mode (PyTriton)

- `GR_RUNTIME_MODE=cuda`
- Guardrails uses PyTriton client.
- PyTriton server hosts GLiNER on GPU and can load it from local `GR_MODEL_DIR`.

## Air-gapped models

Download all required models into a single directory:

```bash
make download-models MODELS_DIR=./.models
```

This writes a model bundle and `manifest.json` into `./.models`.

Run service in offline mode on host:

```bash
make deps-up
uv sync --extra dev --extra eval --extra finetune
GR_MODEL_DIR=./.models GR_OFFLINE_MODE=true GR_REDIS_URL=redis://localhost:6379/0 uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Dataset Evaluation

Manual evaluation framework supports dataset adapters and unified report outputs (JSON + Markdown).

Setup once:

```bash
uv sync --extra dev --extra eval
cp .env.eval.example .env.eval
# set HF_TOKEN in .env.eval
```

Run first dataset evaluation:

```bash
make eval-scanpatch
```

Run all available datasets (default behavior when `--dataset` is omitted):

```bash
uv run --extra eval python -m app.eval.run --split test --policy-path configs/policy.yaml --policy-name external_default --env-file .env.eval --output-dir reports/evaluations
```

Progress/ETA is printed during evaluation (`[progress] ...`). Tune cadence with:
- `--progress-every-samples`
- `--progress-every-seconds`

Run a specific dataset only:

```bash
uv run --extra eval python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test

# dataset without native test split: synthetic split is created and cached
uv run --extra eval python -m app.eval.run --dataset BoburAmirov/rubai-NER-150K-Personal --split test --strict-split
```

Run cascade mode for throughput/quality tradeoff:

```bash
uv run --extra eval python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test --mode cascade --cascade-threshold 0.15 --cascade-heavy-recognizers gliner_pii_multilingual
```

What it does:
- Downloads dataset automatically with Hugging Face token from `.env.eval`.
- Reuses local dataset cache at `.eval_cache/` (no re-download on subsequent runs).
- For datasets without native `test`, builds and caches a synthetic train/test split with label-balancing heuristic.
- Writes report files to `reports/evaluations/`.
- Includes detector-level and dataset-slice metrics in JSON reports.

## GLiNER Fine-tuning

Manual scripts are available for fine-tuning GLiNER on prepared datasets (including train-on-all/eval-on-all MVP feasibility runs).

See:
- `docs/FINETUNING.md`

Quick start:

```bash
make finetune-scanpatch-pipeline
make eval-finetuned-gliner FINETUNE_MODEL_REF=./reports/finetune/scanpatch_pipeline/runs/iter_01/final
```

## Local run

```bash
uv sync --extra dev --extra eval --extra finetune
make deps-up
uv run --extra dev pytest tests/unit -q
# run API (separate terminal)
GR_REDIS_URL=redis://localhost:6379/0 uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Dependencies (Docker Compose)

Compose is used only for infrastructure dependencies on dev machines.

Start Redis:

```bash
docker compose up -d redis
```

Stop dependencies:

```bash
docker compose down --remove-orphans
```

Run PyTriton on host (for `GR_RUNTIME_MODE=cuda`):

```bash
GR_MODEL_DIR=./.models GR_OFFLINE_MODE=true GR_PYTRITON_DEVICE=cuda uv run --extra cuda python -m app.pytriton_server.main
```

Run integration tests:

```bash
make test-integration
```

Equivalent make targets:

```bash
make download-models MODELS_DIR=./.models
make deps-up
make run-api
make run-pytriton
make test-integration
make eval-scanpatch
make eval-scanpatch-baseline
```

`make test-integration` starts the API on host (`localhost:8080`) and runs integration tests against it.

## Key files

- `app/main.py`: API surface
- `app/guardrails.py`: masking/unmasking orchestration
- `app/core/analysis/*`: Presidio analysis backend and recognizers
- `app/runtime/*`: runtime selection and adapters
- `app/pytriton_server/*`: PyTriton model server and model registry
- `app/eval/*`: manual dataset evaluation framework
- `app/finetune/*`: GLiNER data prep, training, and evaluation helpers
- `app/tools/run_scanpatch_gliner_finetune_pipeline.py`: end-to-end tuning pipeline
- `app/tools/evaluate_finetuned_gliner.py`: evaluation-only script for existing GLiNER checkpoints
- `configs/policy.yaml`: policy + analyzer profile + recognizer definitions
- `docs/DETECTORS.md`: recognizer catalog and labels
- `docs/GPU_SUPPORT.md`: PyTriton runtime details
- `docs/EVALUATION.md`: evaluation architecture and report format
- `docs/FINETUNING.md`: GLiNER fine-tuning workflow

## Docker Images

- `Dockerfile`: Guardrails API image (CPU/MPS runtime path inside app).
- `Dockerfile.cuda`: PyTriton CUDA runtime image.
