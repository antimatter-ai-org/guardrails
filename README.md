# Guardrails Service (MVP)

OpenAI-compatible proxy guardrails for on-prem AI platforms.

This service prevents sensitive data leakage to external LLMs by applying **reversible masking** before outbound requests and unmasking on inbound responses.

## What is implemented

- OpenAI-compatible `POST /v1/chat/completions` proxy endpoint.
- Reversible masking pipeline:
  - Detect sensitive spans.
  - Replace with deterministic placeholders (e.g. `<GR00:0001>`).
  - Store placeholder-to-original map in Redis with TTL.
  - Forward masked payload to upstream LLM.
  - Unmask placeholders in LLM response before returning to caller.
- Policy-driven behavior from single YAML file (`configs/policy.yaml`), reloadable via `POST /admin/reload`.
- Model-based routing policy (`external` vs `onprem`) to avoid unnecessary masking for air-gapped models.
- Extendable detector plugins.

## Detector stack (RU/EN focused)

MVP combines several detection approaches:

1. Regex detectors (configurable):
   - Russian PII patterns: phone, passport, SNILS, INN/OGRN, bank card, email.
   - English/common patterns: SSN, IBAN, SWIFT, international phone.
2. Secret detection for code/agent workloads:
   - Signature regexes: AWS keys, GitHub tokens, Slack tokens, JWT, private keys, generic key/value secrets.
3. Entropy detector:
   - Catches high-entropy tokens that often represent unknown secret formats.
4. Russian NER (`natasha`) detector:
   - Detects entities in Russian text with NER.
5. Optional multilingual BERT-like detector (`GLiNER`) plugin:
   - Disabled in default config but implemented and pluggable.

## Why these libraries

- `natasha` (Russian NLP/NER): https://github.com/natasha/natasha
- `GLiNER` (general multilingual transformer-based entity detection): https://github.com/urchade/GLiNER
- We considered Presidio as an orchestration option and kept architecture compatible with that style:
  - https://microsoft.github.io/presidio/

The selected stack gives strong rule-based control for security teams plus ML-assisted recall for Russian and English text.

## Architecture

- `app/main.py`: API endpoints and proxy flow.
- `app/guardrails.py`: masking/unmasking orchestration + policy application.
- `app/masking/engine.py`: span conflict resolution, reversible placeholders.
- `app/masking/payload.py`: OpenAI payload traversal (request/response text slots).
- `app/detectors/*`: detector plugins.
- `app/policy.py` + `app/config.py`: YAML config loading and model-policy resolution.
- `app/storage/redis_store.py`: Redis TTL map store for reversible masking.

## API

- `GET /health`
- `POST /admin/reload`
- `POST /v1/chat/completions` (proxy + guardrails)
- `POST /v1/guardrails/mask` (explicit pre-processing)
- `POST /v1/guardrails/unmask` (explicit post-processing)

## Local run

### 1) Python (unit tests)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest tests/unit -q
```

### 2) Full stack via Docker Compose

```bash
docker compose up -d redis mock-llm guardrails
curl http://localhost:8080/health
```

Run integration tests against full stack:

```bash
docker compose --profile test up --build --abort-on-container-exit --exit-code-from integration-tests integration-tests
```

Or use `Makefile`:

```bash
make test-unit
make test-integration
```

## Example request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-upstream-base-url: http://localhost:8090' \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role":"user","content":"Мой email ivan@example.com и паспорт 1234 567890"}]
  }'
```

Guardrails masks sensitive values before upstream call and restores them in final response.

## Extending detectors

1. Add a detector implementation in `app/detectors/`.
2. Register detector type in `app/detectors/factory.py`.
3. Add a detector definition in `configs/policy.yaml`.
4. Reference detector name in target policy.
5. Reload service (`POST /admin/reload`) or restart.

## MVP limits

- Streaming responses are not yet supported (`stream=true` rejected).
- Placeholder mapping is per request and stored in Redis with TTL.
- No advanced anti-correlation cryptography between requests (intentionally out of MVP scope).
