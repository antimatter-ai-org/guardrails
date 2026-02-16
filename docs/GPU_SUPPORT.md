# GPU Support with Embedded PyTriton

This project has one project-level runtime switch:
- `cpu`: all detectors run in-process via torch (Apple Silicon supported with MPS fallback).
- `cuda`: all supported CUDA-capable models run through PyTriton, started by Guardrails service itself.

Nemotron is controlled by one global flag:
- `GR_ENABLE_NEMOTRON=false` (default)
- set `GR_ENABLE_NEMOTRON=true` to enable Nemotron recognizer and PyTriton model binding.

## Runtime switch

Use environment variable:

- `GR_RUNTIME_MODE=cpu`
- `GR_RUNTIME_MODE=cuda`

There is no per-model backend switch.

## Deployment model

In `cuda` mode:
- Guardrails API process starts embedded PyTriton on startup.
- Guardrails API process stops embedded PyTriton on shutdown.
- Runtime adapters call PyTriton over internal loopback only.
- PyTriton bind address is fixed to `127.0.0.1`.

Important limitation:
- this is still protocol-based Triton inference (HTTP/gRPC on loopback), not Triton C-API in-process invocation.

## Air-gapped model loading

Both guardrails and embedded PyTriton can load models from a local directory:
- set `GR_MODEL_DIR=/path/to/models`
- set `GR_OFFLINE_MODE=true`

## Model behavior

GLiNER recognizer is always enabled by policy. Nemotron recognizer exists in policy but is runtime-gated by `GR_ENABLE_NEMOTRON`.

- In `cpu` mode:
  - Guardrails process loads model runtimes locally with torch/transformers.
  - Device selection uses `GR_CPU_DEVICE` (`auto` by default; prefers `mps` on Apple Silicon, then `cpu`).
  - Model loading is eager at startup (no lazy/background loading).

- In `cuda` mode:
  - Guardrails runtime adapters use PyTriton client.
  - Embedded PyTriton always hosts `gliner`.
  - Embedded PyTriton hosts `nemotron` only when `GR_ENABLE_NEMOTRON=true`.
  - Guardrails startup performs deterministic Triton readiness checks (server live/ready, model ready, I/O contract) and fails fast if any enabled model is not ready.
  - `/readyz` reports ready only after all model runtimes are fully initialized.

## Local runtime usage

Start Redis dependency:

```bash
docker compose up -d redis
```

Run Guardrails in CPU mode:

```bash
GR_RUNTIME_MODE=cpu \
GR_REDIS_URL=redis://localhost:6379/0 \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Run Guardrails in CUDA mode (embedded PyTriton):

```bash
GR_RUNTIME_MODE=cuda \
GR_REDIS_URL=redis://localhost:6379/0 \
uv run --extra cuda uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Optional debug-only standalone PyTriton server:

```bash
make run-pytriton-debug
```

## CUDA image behavior

`Dockerfile.cuda` now runs Guardrails API entrypoint (`uvicorn app.main:app`), not standalone PyTriton.

Externally exposed port:
- `8080` only (Guardrails API).

PyTriton ports are internal-only and bound to loopback in the container.

## Important env vars

Shared:
- `GR_MODEL_DIR=/path/to/models`
- `GR_OFFLINE_MODE=true|false`
- `GR_ENABLE_NEMOTRON=true|false`

CPU mode:
- `GR_RUNTIME_MODE=cpu`
- `GR_CPU_DEVICE=auto`

CUDA mode:
- `GR_RUNTIME_MODE=cuda`
- `GR_PYTRITON_URL=127.0.0.1:8000`
- `GR_PYTRITON_INIT_TIMEOUT_S=30`
- `GR_PYTRITON_INFER_TIMEOUT_S=60`
- `GR_PYTRITON_GLINER_MODEL_REF=urchade/gliner_multi-v2.1`
- `GR_PYTRITON_TOKEN_MODEL_REF=scanpatch/pii-ner-nemotron`
- `GR_PYTRITON_DEVICE=cuda`
- `GR_PYTRITON_MAX_BATCH_SIZE=32`
- `GR_PYTRITON_GRPC_PORT=8001`
- `GR_PYTRITON_METRICS_PORT=8002`

## Pluggability path

- Add model class under `app/pytriton_server/models/`.
- Register binding in `app/pytriton_server/registry.py`.
- Add or extend runtime adapter if guardrails needs model-specific client preprocessing/postprocessing.
