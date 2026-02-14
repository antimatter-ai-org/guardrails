# Guardrails Service (MVP)

Guardrails microservice for LLM routers.

This service does **detection, masking, and unmasking only**. It does not route or proxy model traffic.

## Architecture

Your LLM router owns request forwarding.
This service is called at explicit stages:

1. Router calls `mask` on outbound user content.
2. Router sends masked content to external LLM.
3. Router calls `unmask` for non-streamed responses, or `unmask-stream` per chunk for streamed responses.
4. Router calls `finalize` to cleanup request state.

## Features

- Reversible masking with Redis-backed request context.
- Policy-driven behavior from YAML (`configs/policy.yaml`).
- RU/EN-focused detector stack:
  - Regex PII detectors (Russian + English patterns)
  - Code/secret regex detector
  - Entropy detector
  - Natasha NER (Russian)
  - Optional GLiNER detector plugin (enabled in full ML profile)
- Streaming-safe unmasking with chunk boundary handling.

## API

### `GET /v1/guardrails/policies`
Returns available policy names and default policy.

### `POST /v1/guardrails/detect`
Detection only (no masking, no context storage).

Request:

```json
{
  "policy_name": "external_default",
  "items": [{"id": "msg-1", "text": "Мой email ivan@example.com"}]
}
```

### `POST /v1/guardrails/mask`
Masks sensitive content and optionally stores context for later unmasking.

Request:

```json
{
  "request_id": "req-123",
  "policy_name": "external_default",
  "store_context": true,
  "items": [{"id": "msg-1", "text": "Мой email ivan@example.com"}]
}
```

### `POST /v1/guardrails/unmask`
Unmasks batch text items using stored context.

### `POST /v1/guardrails/unmask-stream`
Unmasks one streamed chunk at a time.

Important fields:
- `stream_id`: logical stream key (for multi-choice/multi-stream outputs).
- `final`: set `true` on last chunk.
- `delete_context`: set `true` on final chunk when request is complete.

### `POST /v1/guardrails/finalize`
Force cleanup of request context and stream buffers.
Use for cancellation/error paths in router.

## Streaming behavior

`unmask-stream` handles placeholders split across chunk boundaries.
It keeps an internal per-stream tail buffer in Redis, so placeholders are only emitted after safe reconstruction.

This avoids leaking raw placeholders when the LLM stream cuts a placeholder token in the middle.

## Policy model

Policies are explicit and selected by router via `policy_name`.
No model routing logic exists in this service.

Policy modes:
- `mask`: mask + store placeholders.
- `passthrough`: leave text unchanged.
- `block`: return 403 from `mask` if sensitive data detected.

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest tests/unit -q
```

For full ML detectors on CPU (includes GLiNER + torch):

```bash
pip install -e '.[dev,ml,ml_torch]'
```

## Docker Compose

Start services:

```bash
docker compose up -d redis guardrails
```

Start full ML service on CPU:

```bash
docker compose --profile ml-cpu up -d redis guardrails-ml-cpu
```

Start GPU service:

```bash
docker compose --profile gpu up -d redis guardrails-gpu
```

Equivalent make targets:

```bash
make dev-up
make dev-up-ml-cpu
make dev-up-gpu
```

Run integration tests:

```bash
docker compose --profile test up --build --abort-on-container-exit --exit-code-from integration-tests integration-tests
```

## Key files

- `app/main.py`: API surface
- `app/guardrails.py`: masking/unmasking orchestration + stream handling
- `app/storage/redis_store.py`: request/stream state in Redis
- `app/detectors/*`: detector plugins
- `configs/policy.yaml`: policy + detector definitions
- `configs/policy.full.yaml`: full-ML policy profile (GLiNER enabled)
- `docs/DETECTORS.md`: detector catalog with labels and examples
- `docs/GPU_SUPPORT.md`: CPU/GPU runtime and deployment options
