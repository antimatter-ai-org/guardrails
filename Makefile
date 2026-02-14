.PHONY: dev-up dev-up-gpu dev-down test-unit test-integration test-all download-models check-models eval-scanpatch eval-scanpatch-baseline eval-scanpatch-cascade

MODELS_DIR ?= ./.models
POLICY_PATH ?= ./configs/policy.yaml
EVAL_ENV_FILE ?= ./.env.eval
EVAL_OUTPUT_DIR ?= ./reports/evaluations

download-models:
	docker compose build guardrails
	docker run --rm -v $(PWD)/$(MODELS_DIR):/models guardrails-guardrails:latest python -m app.tools.download_models --output-dir /models --policy-path $(POLICY_PATH)

check-models:
	@test -f "$(MODELS_DIR)/manifest.json" || (echo "Model bundle not found at $(MODELS_DIR). Run: make download-models MODELS_DIR=$(MODELS_DIR)" && exit 1)

test-unit:
	. .venv/bin/activate && pytest tests/unit -q

dev-up:
	docker compose up -d redis guardrails

dev-up-gpu:
	docker compose --profile gpu up -d redis pytriton guardrails-gpu

dev-down:
	docker compose down --remove-orphans

test-integration: check-models
	GR_MODELS_DIR=$(MODELS_DIR) GR_OFFLINE_MODE=true docker compose --profile test up --build --abort-on-container-exit --exit-code-from integration-tests integration-tests

test-all: test-unit test-integration

eval-scanpatch:
	. .venv/bin/activate && python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test --policy-path $(POLICY_PATH) --policy-name external_default --mode baseline --env-file $(EVAL_ENV_FILE) --output-dir $(EVAL_OUTPUT_DIR)

eval-scanpatch-baseline:
	. .venv/bin/activate && python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test --policy-path $(POLICY_PATH) --policy-name external_default --mode baseline --env-file $(EVAL_ENV_FILE) --output-dir $(EVAL_OUTPUT_DIR)

eval-scanpatch-cascade:
	. .venv/bin/activate && python -m app.eval.run --dataset scanpatch/pii-ner-corpus-synthetic-controlled --split test --policy-path $(POLICY_PATH) --policy-name external_default --mode cascade --cascade-threshold 0.15 --env-file $(EVAL_ENV_FILE) --output-dir $(EVAL_OUTPUT_DIR)
