# Guardrails API Reference

Base path: `/v1/guardrails`

## Model

Guardrails is stage-based and transform-aware.

- `source`: origin of content (`INPUT`, `OUTPUT`, `TOOL_INPUT`, `TOOL_OUTPUT`, `RETRIEVAL`)
- `transforms`: optional reversible operation
- `action`: final decision (`NONE`, `MASKED`, `BLOCKED`, `FLAGGED`)

Configured policy surface for this release:
- Default and only configured policy id: `external`

## Endpoints

### `GET /healthz`
Liveness probe.

### `GET /readyz`
Readiness probe. Returns ready only after dependencies and model runtime warmup pass.

### `GET /openapi.json`
FastAPI-generated OpenAPI spec.

### `GET /v1/guardrails/capabilities`
Returns supported enums, configured policy ids, and recognizer ids.

Example:

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
  "policies": ["external"],
  "checks": [
    "phone_number_lib",
    "ip_address_lib",
    "ru_pii_regex",
    "en_pii_regex",
    "identifier_regex",
    "network_pii_regex",
    "url_regex",
    "date_pii_regex",
    "code_secret_regex",
    "high_entropy_secret",
    "nemotron_pii_token_classifier"
  ],
  "runtime_mode": "cpu"
}
```

### `POST /v1/guardrails/apply`
Unified non-streaming endpoint.

Request shape:

```json
{
  "request_id": "optional-router-request-id",
  "policy_id": "external",
  "source": "INPUT",
  "content": [{"id": "msg-1", "text": "Text to evaluate"}],
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

Notes:
- If `policy_id` is omitted, `external` is used.
- Current implementation supports at most one transform per request.

### `POST /v1/guardrails/apply-stream`
Streaming endpoint for chunk-wise `REIDENTIFY`.

Request shape:

```json
{
  "policy_id": "external",
  "source": "OUTPUT",
  "transforms": [
    {
      "type": "reversible_mask",
      "mode": "REIDENTIFY",
      "session": {"id": "sess_abc123", "allow_missing_context": false}
    }
  ],
  "stream": {
    "id": "choice-0",
    "chunk": "text chunk",
    "final": false
  }
}
```

### `POST /v1/guardrails/sessions/{session_id}/finalize`
Deletes reversible mapping + streaming buffer state for the session.

## Typical Router Flow

1. `apply` with `DEIDENTIFY` on input.
2. Send masked content upstream.
3. Reidentify via:
- `apply` (non-stream), or
- `apply-stream` (streamed chunks).
4. Finalize session via `/sessions/{id}/finalize`.

## Client Generation

```bash
curl -sS http://localhost:8080/openapi.json > openapi.json
```
