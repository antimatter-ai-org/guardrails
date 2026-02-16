# Guardrails v0.0.1

Initial public release of the Guardrails service MVP.

## Highlights

- Unified Guardrails HTTP API for detect/mask/unmask and streaming reidentification.
- Reversible masking flow for external LLM routing safety.
- Presidio-based analyzer stack with regex + GLiNER and optional Nemotron recognizers.
- Streaming-safe placeholder restoration with Redis-backed session state.
- Runtime split by mode:
  - `cpu`: in-process torch inference (Apple Silicon MPS auto path).
  - `cuda`: embedded PyTriton lifecycle managed by the Guardrails service.
- Air-gapped model support with offline model bundle download and local loading.
- Evaluation harness with dataset caching and report generation.

## Release Artifacts

- CPU image: `ghcr.io/antimatter-ai-org/guardrails-cpu:v0.0.1`
- CUDA image: `ghcr.io/antimatter-ai-org/guardrails-cuda:v0.0.1`
