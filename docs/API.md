# Guardrails API Reference

This document describes the unified API surface used by LLM routers.

Base path: `/v1/guardrails`

## Design Model

The API is stage-based and transform-aware:

- `source`: where text came from (`INPUT`, `OUTPUT`, `TOOL_INPUT`, `TOOL_OUTPUT`, `RETRIEVAL`)
- `transforms`: optional reversible transformation directives
  - `DEIDENTIFY`: detect + replace sensitive spans with placeholders
  - `REIDENTIFY`: replace placeholders with original values using session context
- `action`: high-level decision from guardrails (`NONE`, `MASKED`, `BLOCKED`, `FLAGGED`)

## Endpoints

### `GET /openapi.json`

Auto-generated OpenAPI 3.1 JSON spec (FastAPI native endpoint).

### `GET /healthz`

Liveness probe.

Example response:

```json
{
  "status": "ok"
}
```

### `GET /readyz`

Readiness probe. Checks service dependencies and runtime initialization state.
The endpoint returns `ready` only after all configured model runtimes are fully loaded and warm-up is complete.

Example response:

```json
{
  "status": "ready"
}
```

### `GET /v1/guardrails/capabilities`

Returns supported enums and configured checks/policies.

Example response:

```json
{
  "service": "guardrails-service",
  "api_version": "v1",
  "sources": ["INPUT", "OUTPUT", "TOOL_INPUT", "TOOL_OUTPUT", "RETRIEVAL"],
  "actions": ["NONE", "MASKED", "BLOCKED", "FLAGGED"],
  "transforms": ["reversible_mask"],
  "transform_modes": ["DEIDENTIFY", "REIDENTIFY"],
  "output_scopes": ["INTERVENTIONS", "FULL"],
  "trace_levels": ["NONE", "BASIC", "FULL"],
  "policies": ["external_default", "onprem_passthrough", "strict_block"],
  "checks": ["phone_number_lib", "ru_pii_regex", "gliner_pii_multilingual"],
  "runtime_mode": "cpu"
}
```

### `POST /v1/guardrails/apply`

Unified non-streaming endpoint.

#### Request schema

```json
{
  "request_id": "optional-router-request-id",
  "policy_id": "external_default",
  "policy_version": "optional-policy-version-tag",
  "source": "INPUT",
  "content": [
    {
      "id": "msg-1",
      "text": "Text to evaluate"
    }
  ],
  "transforms": [
    {
      "type": "reversible_mask",
      "mode": "DEIDENTIFY",
      "session": {
        "id": "optional-session-id",
        "ttl_seconds": 3600,
        "allow_missing_context": false
      }
    }
  ],
  "output_scope": "FULL",
  "trace": "NONE"
}
```

#### Field semantics

- `policy_id`: policy from `configs/policy.yaml`; if omitted, service default is used.
- `source`: stage marker for auditing and future stage-specific rails.
- `content[]`: batch of text items to process.
- `transforms[]`: max 1 transform per request in current implementation.
- `output_scope`:
  - `INTERVENTIONS`: lean findings payload.
  - `FULL`: includes richer finding evidence metadata.
- `trace`: reserved for future debug/trace level expansion.

#### Response schema

```json
{
  "action": "MASKED",
  "source": "INPUT",
  "policy_id": "external_default",
  "policy_version": null,
  "outputs": [
    {
      "id": "msg-1",
      "text": "Masked or transformed text"
    }
  ],
  "findings": [
    {
      "check_id": "ru_pii_regex:EMAIL_ADDRESS",
      "category": "email",
      "severity": "high",
      "confidence": 0.96,
      "spans": [
        {
          "start": 10,
          "end": 31,
          "snippet": "ivan.petrov@example.com",
          "label": "EMAIL"
        }
      ],
      "evidence": {}
    }
  ],
  "session": {
    "id": "sess_abc123",
    "ttl_seconds": 3600,
    "expires_at": "2026-02-15T22:15:01.123456+00:00"
  },
  "usage": {
    "input_items": 1,
    "input_chars": 52,
    "output_items": 1,
    "output_chars": 41
  },
  "timings": {
    "total_ms": 19.7,
    "detector_timing_ms": {
      "ru_pii_regex:EMAIL_ADDRESS": 1.2,
      "gliner_pii_multilingual": 12.6
    }
  }
}
```

#### Action semantics

- `NONE`: no intervention needed.
- `MASKED`: content was transformed (masked or reidentified replacement happened).
- `BLOCKED`: policy blocked request.
- `FLAGGED`: non-blocking issue detected (for example missing reidentify context when allowed).

### `POST /v1/guardrails/apply-stream`

Streaming endpoint for chunk-wise `REIDENTIFY`.

#### Request schema

```json
{
  "request_id": "optional-router-request-id",
  "policy_id": "external_default",
  "policy_version": null,
  "source": "OUTPUT",
  "transforms": [
    {
      "type": "reversible_mask",
      "mode": "REIDENTIFY",
      "session": {
        "id": "sess_abc123",
        "allow_missing_context": false
      }
    }
  ],
  "output_scope": "FULL",
  "trace": "NONE",
  "stream": {
    "id": "choice-0",
    "chunk": "text chunk from model",
    "final": false
  }
}
```

#### Response schema

```json
{
  "action": "MASKED",
  "source": "OUTPUT",
  "policy_id": "external_default",
  "policy_version": null,
  "stream": {
    "id": "choice-0",
    "chunk": "text chunk from model",
    "final": false
  },
  "output_chunk": "possibly reidentified chunk",
  "replacements": 1,
  "buffered_chars": 6,
  "findings": [],
  "session": null,
  "usage": {
    "input_items": 1,
    "input_chars": 31,
    "output_items": 1,
    "output_chars": 29
  },
  "timings": {
    "total_ms": 0.8,
    "detector_timing_ms": {}
  }
}
```

### `POST /v1/guardrails/sessions/{session_id}/finalize`

Deletes reversible mapping and stream buffer state.

Example response:

```json
{
  "session_id": "sess_abc123",
  "context_deleted": true
}
```

Idempotent: if context is already absent, returns `context_deleted=false`.

## Router Integration Playbooks

### Non-streaming external LLM call

1. Pre-call mask:

```bash
curl -sS http://localhost:8080/v1/guardrails/apply \
  -H "content-type: application/json" \
  -d '{
    "policy_id":"external_default",
    "source":"INPUT",
    "content":[{"id":"u1","text":"Email ivan@example.com"}],
    "transforms":[{"type":"reversible_mask","mode":"DEIDENTIFY"}]
  }'
```

2. Send masked `outputs[].text` to external LLM.
3. Post-call unmask:

```bash
curl -sS http://localhost:8080/v1/guardrails/apply \
  -H "content-type: application/json" \
  -d '{
    "policy_id":"external_default",
    "source":"OUTPUT",
    "content":[{"id":"a1","text":"Model answer with <placeholder>"}],
    "transforms":[{"type":"reversible_mask","mode":"REIDENTIFY","session":{"id":"sess_abc123"}}]
  }'
```

4. Finalize session:

```bash
curl -sS -X POST http://localhost:8080/v1/guardrails/sessions/sess_abc123/finalize
```

### Streaming external LLM call

1. Do input `DEIDENTIFY` once and keep returned `session.id`.
2. For each model chunk, call `apply-stream` with `REIDENTIFY`.
3. Emit `output_chunk` to client as it arrives.
4. When stream ends, send chunk with `final=true`.
5. Call `sessions/{id}/finalize`.

## Generating API Clients

Download spec:

```bash
curl -sS http://localhost:8080/openapi.json > openapi.json
```

Generate TypeScript client (openapi-generator):

```bash
openapi-generator-cli generate \
  -i openapi.json \
  -g typescript-fetch \
  -o ./clients/typescript
```

Generate Go client:

```bash
openapi-generator-cli generate \
  -i openapi.json \
  -g go \
  -o ./clients/go
```

Generate Java client:

```bash
openapi-generator-cli generate \
  -i openapi.json \
  -g java \
  -o ./clients/java
```

## Failure and Edge Cases

- Missing session on `REIDENTIFY`:
  - default (`GR_ALLOW_MISSING_REIDENTIFY_SESSION=false`): `BLOCKED`.
  - if allowed: `FLAGGED` and passthrough content.
- Strict block policy with `DEIDENTIFY`: returns `action=BLOCKED`, no outputs.
- `apply-stream` accepts exactly one transform and currently only `REIDENTIFY`.
- Session TTL expires in Redis according to policy TTL or transform override.

## Notes on Compatibility

- This API is intentionally not tied to OpenAI completion schema.
- It is designed to be router-facing and stage-aware, with a single decision envelope.
