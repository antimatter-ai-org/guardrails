# Guardrails Service

Guardrails microservice for LLM routers.

This service does not route LLM traffic itself. It exposes a unified guardrails API that a router can call before and after model calls.

## Core Features

- Unified `apply` API for stage-based guardrails decisions.
- Reversible masking (`DEIDENTIFY`) and unmasking (`REIDENTIFY`).
- Streaming reidentification (`apply-stream`) with placeholder split safety.
- Redis-backed session storage for reversible mappings and stream buffers.
- RU/EN detector stack (regex + GLiNER + optional Nemotron).
- CPU/CUDA runtime switch:
  - `cpu`: local inference (MPS auto on Apple Silicon when available)
  - `cuda`: embedded PyTriton runtime managed by Guardrails service
- Air-gapped operation with offline model preload.
- Manual evaluation framework with cached datasets/splits.

## API Surface

- `GET /healthz`
- `GET /readyz`
- `GET /openapi.json`
- `POST /admin/reload`
- `GET /v1/guardrails/capabilities`
- `POST /v1/guardrails/apply`
- `POST /v1/guardrails/apply-stream`
- `POST /v1/guardrails/sessions/{session_id}/finalize`

Detailed contract and examples:
- `/Users/oleg/Projects/_antimatter/guardrails/docs/API.md`

Fetch OpenAPI spec for client generation:

```bash
curl -sS http://localhost:8080/openapi.json > openapi.json
```

## Typical Router Flow

1. Call `/v1/guardrails/apply` with `source=INPUT` and `DEIDENTIFY`.
2. Send masked text to external LLM.
3. For non-streaming output, call `/v1/guardrails/apply` with `source=OUTPUT` and `REIDENTIFY`.
4. For streaming output, call `/v1/guardrails/apply-stream` per chunk with `REIDENTIFY`.
5. Call `/v1/guardrails/sessions/{session_id}/finalize` when request completes/cancels/errors.

## Runtime Configuration

Global runtime mode:
- `GR_RUNTIME_MODE=cpu`
- `GR_RUNTIME_MODE=cuda`

CPU mode:
- GLiNER and token-classifier run in-process.
- `GR_CPU_DEVICE=auto` chooses `mps` on Apple Silicon when available, otherwise `cpu`.

CUDA mode:
- Guardrails starts and manages PyTriton during app startup.
- PyTriton binds to loopback (`127.0.0.1`) inside the container and is not router-visible.

Other key settings:
- `GR_ENABLE_NEMOTRON=false` (default)
- `GR_ALLOW_MISSING_REIDENTIFY_SESSION=false` (default fail-closed for missing session)
- `GR_PYTRITON_URL=127.0.0.1:8000` (internal endpoint used by runtime adapters)

## Air-Gapped Models

Download all required models:

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

Run all datasets:

```bash
uv run --extra eval python -m app.eval.run --split test --policy-path configs/policy.yaml --policy-name external_default --env-file .env.eval --output-dir reports/evaluations
```

Run one dataset:

```bash
uv run --extra eval python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test --env-file .env.eval
```

## Local Development

Install dependencies:

```bash
uv sync --extra dev --extra eval
```

Run Redis dependency:

```bash
make deps-up
```

Run tests:

```bash
make test-unit
make test-integration
```

Run API:

```bash
make run-api
```

## Docker

- `Dockerfile`: Guardrails service `cpu` flavor (CPU + Apple Silicon MPS path)
- `Dockerfile.cuda`: Guardrails service `cuda` flavor (embedded PyTriton + CUDA)

`docker-compose.yml` is for dependencies only (Redis).

Suggested image tags:
- `guardrails:cpu`
- `guardrails:cuda`

CPU/MPS container:

```bash
docker build -f Dockerfile -t guardrails:cpu .
docker run --rm -p 8080:8080 \
  -e GR_REDIS_URL=redis://host.docker.internal:6379/0 \
  guardrails:cpu
```

CUDA container:

```bash
docker build -f Dockerfile.cuda -t guardrails:cuda .
docker run --rm --gpus all -p 8080:8080 \
  -e GR_RUNTIME_MODE=cuda \
  -e GR_REDIS_URL=redis://host.docker.internal:6379/0 \
  guardrails:cuda
```

## Router PoC Integration

OpenRouter + LiteLLM integration harness (with Guardrails callback and Postman assets):

- `/Users/oleg/Projects/_antimatter/guardrails/integrations/litellm_openrouter/README.md`

Run with one command:

```bash
docker compose -f integrations/litellm_openrouter/docker-compose.yml --env-file integrations/litellm_openrouter/.env up --build
```

## Key Files

- `/Users/oleg/Projects/_antimatter/guardrails/app/main.py`: HTTP API
- `/Users/oleg/Projects/_antimatter/guardrails/app/guardrails.py`: masking/unmasking orchestration
- `/Users/oleg/Projects/_antimatter/guardrails/app/core/analysis`: detection pipeline and recognizers
- `/Users/oleg/Projects/_antimatter/guardrails/app/core/masking/reversible.py`: reversible masking engine
- `/Users/oleg/Projects/_antimatter/guardrails/app/runtime`: CPU/CUDA runtime adapters
- `/Users/oleg/Projects/_antimatter/guardrails/app/pytriton_server`: PyTriton model server
- `/Users/oleg/Projects/_antimatter/guardrails/app/tools/download_models.py`: model downloader
- `/Users/oleg/Projects/_antimatter/guardrails/app/eval`: evaluation framework
- `/Users/oleg/Projects/_antimatter/guardrails/configs/policy.yaml`: policies/profiles/recognizers
