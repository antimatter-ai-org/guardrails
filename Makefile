.PHONY: sync deps-up deps-down dev-up dev-down run-api run-pytriton test-unit test-integration test-all download-models check-models eval-all eval-scanpatch

MODELS_DIR ?= ./.models
POLICY_PATH ?= ./configs/policy.yaml
EVAL_ENV_FILE ?= ./.env.eval
EVAL_OUTPUT_DIR ?= ./reports/evaluations

sync:
	uv sync --extra dev --extra eval

download-models:
	uv run --extra eval python -m app.tools.download_models --output-dir $(MODELS_DIR) --policy-path $(POLICY_PATH)

check-models:
	@test -f "$(MODELS_DIR)/manifest.json" || (echo "Model bundle not found at $(MODELS_DIR). Run: make download-models MODELS_DIR=$(MODELS_DIR)" && exit 1)

test-unit:
	uv run --extra dev pytest tests/unit -q

dev-up:
	docker compose up -d --remove-orphans redis

deps-up: dev-up

run-api:
	uv run --extra dev uvicorn app.main:app --host 0.0.0.0 --port 8080

run-pytriton:
	uv run --extra cuda python -m app.pytriton_server.main

dev-down:
	docker compose down --remove-orphans

deps-down: dev-down

test-integration: deps-up
	@set -e; \
	uv run --extra dev env GR_REDIS_URL=redis://localhost:6379/0 uvicorn app.main:app --host 127.0.0.1 --port 8080 >/tmp/guardrails-integration.log 2>&1 & \
	APP_PID=$$!; \
	trap 'kill $$APP_PID >/dev/null 2>&1 || true' EXIT; \
	uv run --extra dev env GUARDRAILS_BASE_URL=http://127.0.0.1:8080 pytest tests/integration -q

test-all: test-unit test-integration

eval-all:
	uv run --extra eval python -m app.eval.run --split test --policy-path $(POLICY_PATH) --policy-name external_default --env-file $(EVAL_ENV_FILE) --output-dir $(EVAL_OUTPUT_DIR)

eval-scanpatch:
	uv run --extra eval python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test --policy-path $(POLICY_PATH) --policy-name external_default --env-file $(EVAL_ENV_FILE) --output-dir $(EVAL_OUTPUT_DIR)
