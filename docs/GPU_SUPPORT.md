# GPU Support with PyTriton

This project has one project-level runtime switch:
- `cpu`: all detectors run in-process on CPU (Apple Silicon supported).
- `gpu`: all supported GPU-capable models run through PyTriton.

## Runtime switch

Use environment variable:

- `GR_RUNTIME_MODE=cpu`
- `GR_RUNTIME_MODE=gpu`

There is no per-model backend switch.

## Air-gapped model loading

Both guardrails and PyTriton can load models from a mounted directory:
- set `GR_MODEL_DIR=/models`
- set `GR_OFFLINE_MODE=true`
- mount host directory with downloaded models to `/models`

## GLiNER behavior

GLiNER is enabled by default in `configs/policy.yaml`.

- In `cpu` mode:
  - Guardrails process loads GLiNER locally with torch.
  - Device selection uses `GR_GLINER_CPU_DEVICE` (`cpu` by default, `mps` optional on Apple Silicon).
  - Model loading is lazy/background on first use to keep API startup fast.

- In `gpu` mode:
  - Guardrails process uses PyTriton client.
  - PyTriton server hosts GLiNER model on GPU.

## PyTriton services

PyTriton server entrypoint:
- `python -m app.pytriton_server.main`

Model pipeline today:
- `gliner` (first model)

Pluggability path:
- Add new model class under `app/pytriton_server/models/`.
- Register binding in `app/pytriton_server/registry.py`.
- Add corresponding runtime adapter if guardrails needs model-specific client preprocessing/postprocessing.

## Docker usage

CPU mode:

```bash
docker compose up -d redis guardrails
```

GPU mode with PyTriton:

```bash
docker compose --profile gpu up -d redis pytriton guardrails-gpu
```

## Important env vars

Guardrails CPU:
- `GR_RUNTIME_MODE=cpu`
- `GR_GLINER_CPU_DEVICE=cpu`
- `GR_MODEL_DIR=/models`
- `GR_OFFLINE_MODE=true`

Guardrails GPU:
- `GR_RUNTIME_MODE=gpu`
- `GR_PYTRITON_URL=pytriton:8000`
- `GR_PYTRITON_GLINER_MODEL_NAME=gliner`
- `GR_PYTRITON_INIT_TIMEOUT_S=30`
- `GR_PYTRITON_INFER_TIMEOUT_S=60`
- `GR_MODEL_DIR=/models`
- `GR_OFFLINE_MODE=true`

PyTriton server:
- `GR_PYTRITON_GLINER_MODEL_NAME=gliner`
- `GR_PYTRITON_GLINER_HF_MODEL_NAME=urchade/gliner_multi-v2.1`
- `GR_PYTRITON_DEVICE=cuda`
- `GR_PYTRITON_MAX_BATCH_SIZE=32`
- `GR_MODEL_DIR=/models`
- `GR_OFFLINE_MODE=true`
