.PHONY: dev-up dev-up-gpu dev-down test-unit test-integration test-all download-models

MODELS_DIR ?= ./.models
POLICY_PATH ?= ./configs/policy.yaml

download-models:
	docker compose build guardrails
	docker run --rm -v $(PWD)/$(MODELS_DIR):/models guardrails-guardrails:latest python -m app.tools.download_models --output-dir /models --policy-path $(POLICY_PATH)

test-unit:
	. .venv/bin/activate && pytest tests/unit -q

dev-up:
	docker compose up -d redis guardrails

dev-up-gpu:
	docker compose --profile gpu up -d redis pytriton guardrails-gpu

dev-down:
	docker compose down --remove-orphans

test-integration:
	docker compose --profile test up --build --abort-on-container-exit --exit-code-from integration-tests integration-tests

test-all: test-unit test-integration
