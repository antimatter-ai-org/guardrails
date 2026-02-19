# Guardrails Service

Guardrails microservice for PII detection, reversible masking, and controlled re-identification in LLM traffic.

Release scope after cleanup:
- Single configured policy: `external`
- Single model-backed recognizer: Nemotron token classifier (`nemotron_pii_token_classifier`)
- Legacy GLiNER stack removed from runtime, serving, config, docs, and tests
- Modern eval framework is `app.eval`; legacy module naming is removed

## Quick Start

Install deps:

```bash
uv sync --extra dev --extra eval
```

Run dependencies:

```bash
make deps-up
```

Run API:

```bash
make run-api
```

Run unit tests:

```bash
make test-unit
```

## API Surface

- `GET /healthz`
- `GET /readyz`
- `GET /openapi.json`
- `POST /admin/reload`
- `GET /v1/guardrails/capabilities`
- `POST /v1/guardrails/apply`
- `POST /v1/guardrails/apply-stream`
- `POST /v1/guardrails/sessions/{session_id}/finalize`

Detailed API contract: `docs/API.md`

## Documentation

- Architecture: `docs/ARCHITECTURE.md`
- Operations: `docs/OPERATIONS.md`
- Evaluation: `docs/EVAL.md`
- API contract: `docs/API.md`
- LiteLLM/OpenRouter integration: `integrations/litellm_openrouter/README.md`

## Evaluation Commands

Run suite:

```bash
uv run --extra eval python -m app.eval.cli --suite guardrails_ru --split fast --policy-path configs/policy.yaml --policy-name external --env-file .env.eval --output-dir reports/evaluations
```

Compare two reports:

```bash
uv run --extra eval python -m app.eval.compare --base /path/base.json --new /path/new.json --out /path/diff_dir
```

Merge split-run reports:

```bash
uv run --extra eval python -m app.tools.merge_eval_reports --out /path/merged.json /path/report1.json /path/report2.json
```

## Key Paths

- API app: `app/main.py`
- Guardrails orchestration: `app/guardrails.py`
- Detection pipeline: `app/core/analysis/`
- Runtime adapters: `app/runtime/`
- Embedded Triton server wiring: `app/pytriton_server/`
- Eval framework: `app/eval/`
- Policy config: `configs/policy.yaml`
