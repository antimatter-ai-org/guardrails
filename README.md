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
  - `cpu`: in-process model execution (Apple Silicon supported)
  - `gpu`: model execution through PyTriton

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
- CPU/MPS device for GLiNER is controlled by `GR_GLINER_CPU_DEVICE` (`cpu` by default, `mps` optional).
- GLiNER and Natasha can be loaded from bind-mounted model directory via `GR_MODEL_DIR`.

### GPU mode (PyTriton)

- `GR_RUNTIME_MODE=gpu`
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

GPU mode (PyTriton + guardrails GPU client):

```bash
docker compose --profile gpu up -d redis pytriton guardrails-gpu
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
make dev-up-gpu
make test-integration
```

`make test-integration` uses the local mounted model bundle and offline mode by default, so models are not re-downloaded on each run.

## Key files

- `app/main.py`: API surface
- `app/guardrails.py`: masking/unmasking orchestration
- `app/detectors/*`: detector plugins
- `app/runtime/*`: runtime selection and adapters
- `app/pytriton_server/*`: PyTriton model server and model registry
- `configs/policy.yaml`: policy + detector definitions
- `docs/DETECTORS.md`: detector catalog and labels
- `docs/GPU_SUPPORT.md`: PyTriton runtime details
