.PHONY: sync deps-up deps-down dev-up dev-down run-api run-pytriton test-unit test-integration test-all download-models check-models eval-all eval-scanpatch eval-scanpatch-baseline eval-scanpatch-cascade eval-manifest eval-compare eval-matrix finetune-prepare-scanpatch finetune-scanpatch-pipeline eval-finetuned-gliner

MODELS_DIR ?= ./.models
POLICY_PATH ?= ./configs/policy.yaml
EVAL_ENV_FILE ?= ./.env.eval
EVAL_OUTPUT_DIR ?= ./reports/evaluations
EVAL_BASE_REPORT ?=
EVAL_CANDIDATE_REPORT ?=
EVAL_COMPARISON_OUTPUT ?=
EVAL_POLICY_ARGS ?= --policy-name external_default
EVAL_ABLATION_ARGS ?=
EVAL_RESUME ?= --no-resume
FINETUNE_OUTPUT_DIR ?= ./reports/finetune/scanpatch_pipeline
FINETUNE_MODEL_REF ?= ./reports/finetune/scanpatch_pipeline/runs/iter_01/final

sync:
	uv sync --extra dev --extra eval --extra finetune

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
	uv run --extra eval python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test --policy-path $(POLICY_PATH) --policy-name external_default --mode baseline --env-file $(EVAL_ENV_FILE) --output-dir $(EVAL_OUTPUT_DIR)

eval-scanpatch-baseline:
	uv run --extra eval python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test --policy-path $(POLICY_PATH) --policy-name external_default --mode baseline --env-file $(EVAL_ENV_FILE) --output-dir $(EVAL_OUTPUT_DIR)

eval-scanpatch-cascade:
	uv run --extra eval python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test --policy-path $(POLICY_PATH) --policy-name external_default --mode cascade --cascade-threshold 0.15 --cascade-heavy-recognizers gliner_pii_multilingual --env-file $(EVAL_ENV_FILE) --output-dir $(EVAL_OUTPUT_DIR)

eval-manifest:
	uv run --extra eval python -m app.tools.create_eval_manifest --report $(EVAL_BASE_REPORT) --output $(EVAL_OUTPUT_DIR)/baseline_manifest.json

eval-compare:
	uv run --extra eval python -m app.tools.compare_eval_reports --base $(EVAL_BASE_REPORT) --candidate $(EVAL_CANDIDATE_REPORT) $(if $(EVAL_COMPARISON_OUTPUT),--output $(EVAL_COMPARISON_OUTPUT),)

eval-matrix:
	uv run --extra eval python -m app.tools.eval_matrix --policy-path $(POLICY_PATH) $(EVAL_POLICY_ARGS) $(EVAL_ABLATION_ARGS) $(EVAL_RESUME) --env-file $(EVAL_ENV_FILE) --output-dir $(EVAL_OUTPUT_DIR) $(if $(EVAL_COMPARISON_OUTPUT),--comparison-output $(EVAL_COMPARISON_OUTPUT),)

finetune-prepare-scanpatch:
	uv run --extra eval python -m app.tools.prepare_gliner_scanpatch_data --dataset scanpatch/pii-ner-corpus-synthetic-controlled --env-file $(EVAL_ENV_FILE)

finetune-scanpatch-pipeline:
	uv run --extra eval --extra finetune python -m app.tools.run_scanpatch_gliner_finetune_pipeline --dataset scanpatch/pii-ner-corpus-synthetic-controlled --env-file $(EVAL_ENV_FILE) --output-dir $(FINETUNE_OUTPUT_DIR)

eval-finetuned-gliner:
	uv run --extra eval python -m app.tools.evaluate_finetuned_gliner --model-ref $(FINETUNE_MODEL_REF) --dataset scanpatch/pii-ner-corpus-synthetic-controlled --env-file $(EVAL_ENV_FILE) --output-dir $(FINETUNE_OUTPUT_DIR) --flat-ner --skip-overlap-metrics --skip-per-label-metrics
