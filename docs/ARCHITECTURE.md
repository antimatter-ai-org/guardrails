# Architecture

## Scope

Current release architecture is intentionally constrained:
- One policy id: `external`
- One model recognizer family: Nemotron token classifier
- No GLiNER runtime or serving components

## Service Layers

- API layer: `app/main.py`
- Guardrails orchestration: `app/guardrails.py`
- Policy resolution: `app/policy.py`
- Detection pipeline (Presidio + recognizers): `app/core/analysis/`
- Reversible masking: `app/core/masking/reversible.py`
- Storage: Redis mapping store in `app/storage/redis_store.py`
- Runtime adapters: `app/runtime/`
- Embedded Triton management: `app/runtime/pytriton_embedded.py`
- Triton model binding registry: `app/pytriton_server/registry.py`

## Policy Model

Policy source of truth is `configs/policy.yaml`.

- `default_policy`: `external`
- `policies`: only `external`
- `analyzer_profiles`: only `external`

Configured recognizer order under `external` profile:
- `phone_number_lib`
- `ip_address_lib`
- `ru_pii_regex`
- `en_pii_regex`
- `identifier_regex`
- `network_pii_regex`
- `url_regex`
- `date_pii_regex`
- `code_secret_regex`
- `high_entropy_secret`
- `nemotron_pii_token_classifier`

## Recognizer Inventory

- `phone_number_lib`: libphonenumber-based detector
- `ip_address_lib`: IP address parser/validator detector
- `ru_pii_regex`: RU-centric regex set
- `en_pii_regex`: EN-centric regex set
- `identifier_regex`: document/identifier regex set
- `network_pii_regex`: network address regex set
- `url_regex`: URL/domain regex set
- `date_pii_regex`: date regex set
- `code_secret_regex`: secret/token regex set
- `high_entropy_secret`: entropy-based secret detector
- `nemotron_pii_token_classifier`: token-classifier recognizer backed by `scanpatch/pii-ner-nemotron`

Regex/secret pattern labels present in config:
- `PHONE_NUMBER`
- `DOCUMENT_NUMBER`
- `TIN`
- `CREDIT_CARD`
- `EMAIL_ADDRESS`
- `US_SSN`
- `IBAN_CODE`
- `SWIFT_CODE`
- `MILITARY_INDIVIDUAL_NUMBER`
- `VEHICLE_NUMBER`
- `IP_ADDRESS`
- `URL`
- `DATE_TIME`
- `API_KEY`

## Runtime Behavior

### CPU
- Detection runs in-process.
- Token classifier uses local `transformers` runtime.

### CUDA
- Guardrails starts embedded PyTriton.
- Embedded binding includes Nemotron model only.
- Readiness requires model contract validation before `/readyz` passes.

## Reversible Masking

- `DEIDENTIFY` stores placeholder-to-original mapping in Redis.
- `REIDENTIFY` restores values from session context.
- Stream mode supports split placeholders via buffered chunk state.
- Session finalization explicitly clears mapping + stream state.
