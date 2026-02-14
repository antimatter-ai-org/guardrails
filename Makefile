.PHONY: dev-up dev-up-ml-cpu dev-up-gpu dev-down test-unit test-integration test-all

test-unit:
	. .venv/bin/activate && pytest tests/unit -q

dev-up:
	docker compose up -d redis guardrails

dev-up-ml-cpu:
	docker compose --profile ml-cpu up -d redis guardrails-ml-cpu

dev-up-gpu:
	docker compose --profile gpu up -d redis guardrails-gpu

dev-down:
	docker compose down --remove-orphans

test-integration:
	docker compose --profile test up --build --abort-on-container-exit --exit-code-from integration-tests integration-tests

test-all: test-unit test-integration
