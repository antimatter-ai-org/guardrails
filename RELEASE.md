# Guardrails Releases

## v0.0.2

Release date: 2026-02-19

Changes from `v0.0.1`:
- 67 commits
- 129 files changed
- 11,918 insertions / 4,339 deletions

### Highlights

- Nemotron-only runtime stack:
  - Removed GLiNER runtime/model-serving code paths.
  - Kept a single active policy: `external`.
- Policy/config simplification:
  - Removed legacy policies (`external_default`, `onprem_passthrough`, `strict_block`).
  - Unified defaults and capabilities around `external`.
- Evaluation framework modernization:
  - Promoted the suite-based evaluator to canonical `app.eval.*`.
  - Removed legacy evaluator paths and old merge tooling.
  - Added robust report merge/compare flow and improved run-id determinism.
- Runtime reliability and performance hardening:
  - Added Triton readiness checks and stricter runtime startup behavior.
  - Improved tokenizer/runtime handling (including Mistral tokenizer fix path).
  - Strengthened span normalization/masking behavior for better safety consistency.
- Detection/tuning improvements:
  - Added URL and extended secret detectors and refined secret pattern gates.
  - Tuned Nemotron thresholds and aggregation strategy.
- LiteLLM/OpenRouter integration:
  - Added full integration harness (router callback, compose stack, Postman/Bruno assets).
  - Updated integration defaults to `GUARDRAILS_POLICY_ID=external` and Nemotron-enabled flow.
- Tooling and operations:
  - Added remote eval scripts and dataset publishing/rebalancing helpers.
  - Expanded long-context dataset generation tooling.
- Documentation cleanup and consolidation:
  - Reworked docs into focused architecture/eval/operations references.
  - Removed redundant legacy detector/eval docs and stale decision/report material.

### Breaking Changes

- Policy IDs changed:
  - `external_default` removed
  - `external_nemotron_only` replaced by `external`
  - `onprem_passthrough` removed
  - `strict_block` removed
- GLiNER-specific runtime/config surface removed.
- Eval module paths changed from `app.eval_v3.*` to `app.eval.*`.
- Eval merge tool changed from `app.tools.merge_eval_v3_reports` to `app.tools.merge_eval_reports`.
- Backward-compatibility shims for removed policy/model/eval surfaces are not preserved.

### Migration Notes

- Use policy `external` in API requests and integration callbacks.
- Use:
  - `python -m app.eval.cli`
  - `python -m app.eval.compare`
  - `python -m app.tools.merge_eval_reports`
- For LiteLLM demo environments, ensure:
  - `GUARDRAILS_POLICY_ID=external`
  - `GR_ENABLE_NEMOTRON=true`

## v0.0.1

Initial public release of the Guardrails service MVP.

### Highlights

- Unified Guardrails HTTP API for detect/mask/unmask and streaming reidentification.
- Reversible masking flow for external LLM routing safety.
- Presidio-based analyzer stack with regex + GLiNER and optional Nemotron recognizers.
- Streaming-safe placeholder restoration with Redis-backed session state.
- Runtime split by mode:
  - `cpu`: in-process torch inference (Apple Silicon MPS auto path).
  - `cuda`: embedded PyTriton lifecycle managed by the Guardrails service.
- Air-gapped model support with offline model bundle download and local loading.
- Evaluation harness with dataset caching and report generation.

### Release Artifacts

- CPU image: `ghcr.io/antimatter-ai-org/guardrails-cpu:v0.0.1`
- CUDA image: `ghcr.io/antimatter-ai-org/guardrails-cuda:v0.0.1`
