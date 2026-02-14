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
- GLiNER model is loaded lazily in background on first use; early requests may run without GLiNER until warm-up completes.

### GPU mode (PyTriton)

- `GR_RUNTIME_MODE=gpu`
- Guardrails uses PyTriton client.
- PyTriton server hosts GLiNER on GPU.

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

Run integration tests:

```bash
docker compose --profile test up --build --abort-on-container-exit --exit-code-from integration-tests integration-tests
```

Equivalent make targets:

```bash
make dev-up
make dev-up-gpu
```

## Key files

- `app/main.py`: API surface
- `app/guardrails.py`: masking/unmasking orchestration
- `app/detectors/*`: detector plugins
- `app/runtime/*`: runtime selection and adapters
- `app/pytriton_server/*`: PyTriton model server and model registry
- `configs/policy.yaml`: policy + detector definitions
- `docs/DETECTORS.md`: detector catalog and labels
- `docs/GPU_SUPPORT.md`: PyTriton runtime details
