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
- RU/EN detector stack, including GLiNER enabled by default.
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
- GLiNER and Natasha can be loaded from bind-mounted model directory via `GR_MODEL_DIR`.

### CUDA mode (PyTriton)

- `GR_RUNTIME_MODE=cuda`
- Guardrails uses PyTriton client.
- PyTriton server hosts GLiNER on GPU and can load it from `GR_MODEL_DIR`.

## Air-gapped models

Download all required models (GLiNER + Natasha) into a single directory:

```bash
make download-models MODELS_DIR=./.models
```

This writes a model bundle and `manifest.json` into `./.models`.

Run service in offline mode with mounted models:

```bash
GR_MODELS_DIR=./.models GR_OFFLINE_MODE=true docker compose up -d redis guardrails
```

## Dataset Evaluation

Manual evaluation framework supports dataset adapters and unified report outputs (JSON + Markdown).

Setup once:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[ml,eval,dev]'
cp .env.eval.example .env.eval
# set HF_TOKEN in .env.eval
```

Run first dataset evaluation:

```bash
make eval-scanpatch
```

Run cascade mode for throughput/quality tradeoff:

```bash
python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test --mode cascade --cascade-threshold 0.15
```

What it does:
- Downloads dataset automatically with Hugging Face token from `.env.eval`.
- Reuses local dataset cache at `.eval_cache/` (no re-download on subsequent runs).
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
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,ml]'
pytest tests/unit -q
```

## Docker Compose

CPU mode:

```bash
docker compose up -d redis guardrails
```

CUDA mode (PyTriton + guardrails CUDA client):

```bash
docker compose --profile cuda up -d redis pytriton guardrails-cuda
```

Offline mode (no HF network fetches, models from mount):

```bash
GR_MODELS_DIR=./.models GR_OFFLINE_MODE=true docker compose up -d redis guardrails
```

Run integration tests:

```bash
GR_MODELS_DIR=./.models GR_OFFLINE_MODE=true docker compose --profile test up --build --abort-on-container-exit --exit-code-from integration-tests integration-tests
```

Equivalent make targets:

```bash
make download-models MODELS_DIR=./.models
make dev-up
make dev-up-cuda
make test-integration
make eval-scanpatch
make eval-scanpatch-baseline
```

`make test-integration` uses the local mounted model bundle and offline mode by default, so models are not re-downloaded on each run.

## Key files

- `app/main.py`: API surface
- `app/guardrails.py`: masking/unmasking orchestration
- `app/detectors/*`: detector plugins
- `app/runtime/*`: runtime selection and adapters
- `app/pytriton_server/*`: PyTriton model server and model registry
- `app/eval/*`: manual dataset evaluation framework
- `app/finetune/*`: GLiNER data prep, training, and evaluation helpers
- `app/tools/run_scanpatch_gliner_finetune_pipeline.py`: end-to-end tuning pipeline
- `app/tools/evaluate_finetuned_gliner.py`: evaluation-only script for existing GLiNER checkpoints
- `configs/policy.yaml`: policy + detector definitions
- `docs/DETECTORS.md`: detector catalog and labels
- `docs/GPU_SUPPORT.md`: PyTriton runtime details
- `docs/EVALUATION.md`: evaluation architecture and report format
- `docs/FINETUNING.md`: GLiNER fine-tuning workflow
