# GPU Support and Model Runtime

This project supports both CPU-only and GPU-accelerated detector execution.

Goals:
- CPU-first developer experience (including Apple Silicon machines).
- Optional GPU acceleration for transformer-based detectors (for example GLiNER).
- Runtime architecture that can be extended to additional model backends in the future.

## Current runtime architecture

GLiNER detector uses a dedicated runtime layer:
- `app/runtime/torch_runtime.py`: device resolution (`auto`, `cpu`, `cuda`, `mps`)
- `app/runtime/gliner_runtime.py`: GLiNER runtime backend factory

Supported GLiNER backend today:
- `local_torch` (default)

Future extension point:
- Add backend adapters in `build_gliner_runtime(...)` for external model serving systems.

## Device selection

Config knobs:
- Detector params in policy YAML:
  - `backend` (default: `local_torch`)
  - `device` (default: `auto`)
  - `use_fp16_on_cuda` (default: `false`)
- Environment fallbacks:
  - `GR_GLINER_BACKEND`
  - `GR_GLINER_DEVICE`
  - `GR_GLINER_USE_FP16_ON_CUDA`

Device behavior (`auto`):
1. Use CUDA if available.
2. Else use Apple MPS if available.
3. Else use CPU.

## Docker variants

### Baseline CPU service (no heavy ML deps)

```bash
docker compose up -d redis guardrails
```

- Uses `Dockerfile`
- Uses `configs/policy.yaml`
- GLiNER remains disabled by default.

### Full CPU service (all detectors, no GPU required)

```bash
docker compose --profile ml-cpu up -d redis guardrails-ml-cpu
```

- Uses `Dockerfile.ml`
- Uses `configs/policy.full.yaml`
- Installs GLiNER + Torch CPU path
- Binds service to `localhost:8081`

### GPU service

```bash
docker compose --profile gpu up -d redis guardrails-gpu
```

- Uses `Dockerfile.gpu`
- Uses `configs/policy.full.yaml`
- Requests `gpus: all`
- Sets `GR_GLINER_DEVICE=cuda`
- Binds service to `localhost:8082`

## Why separate images

We keep baseline CPU image lightweight and deterministic.
GPU-specific dependencies are isolated in a dedicated image so CPU environments stay fast and simple.

## Triton and alternatives

Triton is a good option when you need multi-model scheduling and batching on shared GPUs.
Other viable choices depending on your constraints:
- Ray Serve: good Python-native serving/orchestration for mixed model graphs.
- KServe: Kubernetes-native model serving with standardized inference APIs.
- vLLM: strong for LLM serving specifically (less focused on arbitrary small NLP classifiers).
- Custom FastAPI + Torch runtime (current approach): simplest operationally for a small number of detectors.

References:
- Triton dynamic batching and concurrent model execution: https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2540/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html
- Ray Serve docs: https://docs.ray.io/en/master/serve/index.html
- KServe docs: https://kserve.github.io/website/0.7/
- vLLM docs: https://docs.vllm.ai/en/v0.6.0/

Recommended path for this project now:
1. Keep `local_torch` runtime as default.
2. Add an optional remote backend adapter later (for example Triton HTTP/gRPC) once detector fleet grows.
3. Route only heavy detectors to GPU backend while keeping regex/rule detectors in-process on CPU.
