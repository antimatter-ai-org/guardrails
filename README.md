# Guardrails Service

Guardrails microservice for LLM routers.

This project provides detection, reversible masking, and unmasking APIs. It does not implement LLM routing.

## Core Features

- Reversible masking for request payloads.
- Response unmasking for full and streaming outputs.
- Redis-backed request/stream state.
- RU/EN recognizer stack with GLiNER + Nemotron token-classifier + deterministic recognizers.
- Language-agnostic runtime analysis (no language hints or language routing in API/runtime).
- Project-level runtime switch:
  - `cpu`: local inference (auto-uses MPS on Apple Silicon when available)
  - `cuda`: all supported ML detectors run via PyTriton
- Single Nemotron knob:
  - `GR_ENABLE_NEMOTRON=false` by default
  - set `GR_ENABLE_NEMOTRON=true` to enable Nemotron detector path
- Air-gapped model workflow:
  - pre-download models with one command
  - run service with offline flags and local model directory
- Manual evaluation harness:
  - runs one dataset or all datasets
  - caches downloaded datasets
  - creates and caches synthetic test split for datasets without native test split
  - unified JSON + Markdown report output

## API

- `GET /health`
- `GET /v1/guardrails/policies`
- `POST /v1/guardrails/detect`
- `POST /v1/guardrails/mask`
- `POST /v1/guardrails/unmask`
- `POST /v1/guardrails/unmask-stream`
- `POST /v1/guardrails/finalize`

Router call sequence:
1. Call `mask`.
2. Send masked text to external LLM.
3. Call `unmask` or `unmask-stream`.
4. Call `finalize` on completion/cancel/error.

## Runtime

Global runtime mode:
- `GR_RUNTIME_MODE=cpu`
- `GR_RUNTIME_MODE=cuda`

CPU mode:
- GLiNER and token-classifier detectors run in-process.
- `GR_CPU_DEVICE=auto` selects `mps` on Apple Silicon when available, otherwise `cpu`.

CUDA mode:
- Guardrails uses PyTriton client.
- Run PyTriton server separately (`python -m app.pytriton_server.main`).
- If using Nemotron, set `GR_ENABLE_NEMOTRON=true` in both guardrails and PyTriton environments.

## Air-Gapped Models

Download all required models from policy:

```bash
make download-models MODELS_DIR=./.models
```

Run offline:

```bash
GR_MODEL_DIR=./.models \
GR_OFFLINE_MODE=true \
GR_REDIS_URL=redis://localhost:6379/0 \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Evaluation

Setup:

```bash
uv sync --extra dev --extra eval
cp .env.eval.example .env.eval
# set HF_TOKEN in .env.eval
```

Run all datasets (default):

```bash
uv run --extra eval python -m app.eval.run --split test --policy-path configs/policy.yaml --policy-name external_default --env-file .env.eval --output-dir reports/evaluations
```

Run one dataset:

```bash
uv run --extra eval python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test --env-file .env.eval
```

Progress is printed during execution (`[progress] ...`).  
Synthetic test split is auto-created and cached for datasets without native `test`.

## Local Development

Install:

```bash
uv sync --extra dev --extra eval
```

Start dependency:

```bash
make deps-up
```

Run unit tests:

```bash
make test-unit
```

Run integration tests:

```bash
make test-integration
```

Run API:

```bash
make run-api
```

Run PyTriton server (CUDA hosts):

```bash
make run-pytriton
```

## Docker

- `Dockerfile`: Guardrails API image.
- `Dockerfile.cuda`: PyTriton CUDA image.

Docker Compose in this repo is for dependencies only (`redis`).

## Key Files

- `app/main.py`: HTTP API
- `app/guardrails.py`: masking/unmasking orchestration
- `app/core/analysis/*`: detection pipeline and recognizers
- `app/core/masking/reversible.py`: masking engine
- `app/runtime/*`: CPU/CUDA runtime adapters
- `app/pytriton_server/*`: PyTriton server
- `app/tools/download_models.py`: model bundle downloader
- `app/eval/*`: evaluation framework
- `configs/policy.yaml`: policies/profiles/recognizers
