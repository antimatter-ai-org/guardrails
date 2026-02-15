# GPU Support with PyTriton

This project has one project-level runtime switch:
- `cpu`: all detectors run in-process via torch (Apple Silicon supported with MPS fallback).
- `cuda`: all supported CUDA-capable models run through PyTriton.

## Runtime switch

Use environment variable:

- `GR_RUNTIME_MODE=cpu`
- `GR_RUNTIME_MODE=cuda`

There is no per-model backend switch.

## Air-gapped model loading

Both guardrails and PyTriton can load models from a local directory:
- set `GR_MODEL_DIR=/path/to/models`
- set `GR_OFFLINE_MODE=true`

## Model behavior

GLiNER and Nemotron token-classifier recognizers are enabled in `configs/policy.yaml`.

- In `cpu` mode:
  - Guardrails process loads model runtimes locally with torch/transformers.
  - Device selection uses `GR_CPU_DEVICE` (`auto` by default; prefers `mps` on Apple Silicon, then `cpu`).
  - Model loading is lazy/background on first use to keep API startup fast.

- In `cuda` mode:
  - Guardrails process uses PyTriton client.
  - PyTriton server hosts all supported ML models on CUDA GPU (`gliner`, `nemotron`).

## PyTriton services

PyTriton server entrypoint:
- `python -m app.pytriton_server.main`

Model pipeline today:
- `gliner`
- `nemotron` (scanpatch/pii-ner-nemotron token-classifier)

Pluggability path:
- Add new model class under `app/pytriton_server/models/`.
- Register binding in `app/pytriton_server/registry.py`.
- Add corresponding runtime adapter if guardrails needs model-specific client preprocessing/postprocessing.

## Local runtime usage

Compose dependencies:

```bash
docker compose up -d redis
```

CUDA mode with PyTriton:

```bash
uv run --extra cuda python -m app.pytriton_server.main
```

Guardrails API (host process):

```bash
GR_REDIS_URL=redis://localhost:6379/0 uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Important env vars

Guardrails CPU:
- `GR_RUNTIME_MODE=cpu`
- `GR_CPU_DEVICE=auto`
- `GR_MODEL_DIR=/path/to/models`
- `GR_OFFLINE_MODE=true`

Guardrails CUDA:
- `GR_RUNTIME_MODE=cuda`
- `GR_PYTRITON_URL=localhost:8000`
- `GR_PYTRITON_INIT_TIMEOUT_S=30`
- `GR_PYTRITON_INFER_TIMEOUT_S=60`
- `GR_MODEL_DIR=/path/to/models`
- `GR_OFFLINE_MODE=true`

PyTriton server:
- `GR_PYTRITON_GLINER_MODEL_REF=urchade/gliner_multi-v2.1`
- `GR_PYTRITON_TOKEN_MODEL_REF=scanpatch/pii-ner-nemotron`
- `GR_PYTRITON_DEVICE=cuda`
- `GR_PYTRITON_MAX_BATCH_SIZE=32`
- `GR_MODEL_DIR=/path/to/models`
- `GR_OFFLINE_MODE=true`
