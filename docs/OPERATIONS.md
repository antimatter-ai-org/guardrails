# Operations

## Required Services

- Guardrails API container/process
- Redis (`GR_REDIS_URL`)

## Key Environment Variables

Core:
- `GR_POLICY_PATH` (default `configs/policy.yaml`)
- `GR_REDIS_URL`
- `GR_RUNTIME_MODE` (`cpu` or `cuda`)
- `GR_ENABLE_NEMOTRON` (default `true`)

Runtime/model:
- `GR_MODEL_DIR` (optional local model bundle path)
- `GR_OFFLINE_MODE` (`true` for offline model usage)
- `GR_CPU_DEVICE` (`auto`, `cpu`, `mps`)

Embedded Triton (CUDA mode):
- `GR_PYTRITON_URL` (loopback only)
- `GR_PYTRITON_GRPC_PORT`
- `GR_PYTRITON_METRICS_PORT`
- `GR_PYTRITON_MAX_BATCH_SIZE`
- `GR_PYTRITON_TOKEN_MODEL_REF`

Timeouts:
- `GR_PYTRITON_INIT_TIMEOUT_S`
- `GR_PYTRITON_INFER_TIMEOUT_S`

## Run Modes

### CPU

```bash
GR_RUNTIME_MODE=cpu uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### CUDA

```bash
GR_RUNTIME_MODE=cuda uv run --extra cuda uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Preloading Assets

Models:

```bash
make download-models MODELS_DIR=./.models POLICY_PATH=./configs/policy.yaml
```

Datasets:

```bash
make download-datasets DATASETS_DIR=./.datasets
```

## Offline Operation

```bash
GR_MODEL_DIR=./.models \
GR_OFFLINE_MODE=true \
GR_RUNTIME_MODE=cpu \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Health and Readiness

- Liveness: `GET /healthz`
- Readiness: `GET /readyz`

`/readyz` remains non-ready until:
- Redis responds
- runtime initialization succeeds
- model readiness checks pass

## Remote Eval Helpers

- SSH helper: `scripts/remote_eval/ssh_eval.sh`
- K8s helper: `scripts/remote_eval/k8s_eval.sh`

Both use `app.eval.cli` and produce report artifacts under `reports/evaluations/`.
