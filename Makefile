.PHONY: dev-up dev-up-gpu dev-down test-unit test-integration test-all download-models check-models

MODELS_DIR ?= ./.models
POLICY_PATH ?= ./configs/policy.yaml

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
