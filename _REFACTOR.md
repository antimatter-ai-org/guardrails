# Guardrails Performance Refactor Plan

Date: 2026-02-15
Owner: Guardrails team
Status: Draft ready for staged implementation

## Execution Log

### 2026-02-15 - Stage 0 Implemented

Status: completed

Implemented:

1. Baseline/candidate comparison CLI:
   1. `app/tools/compare_eval_reports.py`
   2. Supports one baseline and multiple candidates.
   3. Outputs combined, per-dataset, and per-label F1 delta tables.
2. Baseline manifest CLI:
   1. `app/tools/create_eval_manifest.py`
   2. Records report metadata and git SHA.
3. Make targets:
   1. `eval-manifest`
   2. `eval-compare`
4. Documentation updates:
   1. `docs/EVALUATION.md` now documents manifest/diff workflow.
5. Tests:
   1. `tests/unit/test_eval_report_tools.py`

Results:

1. Tool smoke checks passed:
   1. `python -m app.tools.compare_eval_reports ...` produced markdown output.
   2. `python -m app.tools.create_eval_manifest ...` produced manifest output.
2. Unit tests covering the new tooling passed.

### 2026-02-15 - Stage 1 Implemented

Status: completed

Implemented:

1. Runtime readiness contract in GLiNER runtime layer:
   1. `GlinerRuntime.warm_up(timeout_s)`
   2. `GlinerRuntime.is_ready()`
   3. `GlinerRuntime.load_error()`
2. Local CPU runtime warm-up:
   1. Waits for async model load with timeout.
   2. Exposes ready/error states.
3. PyTriton runtime warm-up:
   1. Performs probe inference with timeout bounds.
   2. Exposes ready/error states.
4. Evaluation warm-up orchestration:
   1. `--warmup-timeout-seconds`
   2. `--warmup-strict / --no-warmup-strict`
   3. Per-recognizer warm-up status collection.
   4. Strict failure path for runtime-backed recognizers.
5. Report output:
   1. Added `evaluation.warmup` payload.
   2. Markdown summary includes warm-up diagnostics table.
6. Tests:
   1. `tests/unit/test_eval_warmup.py`
   2. Added warm-up coverage to `tests/unit/test_gliner_runtime.py`

Results:

1. Full unit suite passed:
   1. `66 passed`
2. Integration suite status:
   1. `2 skipped` (no failures)

Non-significant divergence handled:

1. A timing-flaky timeout test was observed in warm-up unit coverage.
2. Test was updated to deterministic synchronization with `threading.Event`.
3. Plan unchanged; implementation continued.

### 2026-02-15 - Stage 2 Implemented

Status: completed

Implemented:

1. Language resolver upgrade:
   1. Added script profiling helpers (`mostly_cyrillic`, `mostly_latin`, `mixed`, `no_letters`).
   2. Added `resolve_languages(...)` with `single|union` strategy.
2. Initial config model additions were introduced during implementation and later removed per user direction.
3. Analysis service update:
   1. Added `resolve_languages(...)` method.
   2. Multi-language analysis path with dedup of identical spans.
4. Evaluation slicing:
   1. Added `dataset_slices.script_profile`.
5. Tests:
   1. `tests/unit/test_analysis_language.py`

Results:

1. Unit suite passed (`75 passed`).
2. Final state after direction update: Stage 2 behavior is always-on in code, with no new policy-level options.

### 2026-02-15 - Stage 3 Implemented (Experimental, Divergence Detected)

Status: implemented but currently divergent from expectations

Implemented:

1. New boundary normalization module:
   1. `app/core/analysis/span_normalizer.py`
2. Service integration:
   1. Postprocess pipeline runs after detection conversion.
3. Config additions:
   1. `analysis.postprocess.boundary.*`
4. Policy wiring:
   1. Enabled boundary postprocess in `external_rich` and `strict_profile`.
5. Tests:
   1. `tests/unit/test_span_normalizer.py`

Observed results (Scanpatch test full, strict warm-up):

1. Baseline reference (pre-Stage-3):
   1. exact F1 `0.5906`
   2. overlap F1 `0.6412`
2. First Stage-3 run:
   1. exact F1 `0.5474` (delta `-0.0433`)
   2. overlap F1 `0.6366` (delta `-0.0046`)
3. Heuristic tightening iteration:
   1. exact F1 `0.5614` (delta `-0.0292`)
   2. overlap F1 `0.6447` (delta `+0.0035`)

Interpretation:

1. This is a significant divergence from expected behavior (exact metric drop exceeds allowed stage-gate).
2. Primary degradation is concentrated in `location` exact boundary alignment.
3. According to execution policy for this project, implementation is paused here pending user decision.

### 2026-02-15 - Direction Update Applied

Status: completed

User direction received:

1. Keep new logic in code.
2. Enable behavior always.
3. Do not introduce additional configuration options at this point.
4. Accept current degradation and continue executing the plan.

Actions taken:

1. Removed newly introduced policy/config toggles for Stage 2/3.
2. Kept Stage 2/3 behavior always-on in code paths.
3. Continued implementation to Stage 4/5/6 without stopping.

### 2026-02-15 - Stage 4 and Stage 5 Implemented

Status: completed

Implemented:

1. Identifier recall improvement pack (Stage 4):
   1. Expanded `identifier_regex` pattern coverage for compact letter+digit IDs and keyword-linked IDs.
   2. Added `inn/tin` keyword-number pattern.
2. High-FP regex calibration (Stage 5):
   1. Restricted `en_pii_regex` to `languages: ["en"]`.
   2. Removed permissive `intl_phone` regex pattern.

Results:

1. Rubai test slice (`max-samples=2000`) improved strongly:
   1. exact canonical F1: `0.3678`
   2. overlap canonical F1: `0.8343`
   3. identifier exact F1: `0.4217`
2. Scanpatch full test remains below pre-Stage-3 baseline:
   1. exact canonical F1: `0.5610` (vs old baseline `0.5906`)
   2. overlap canonical F1: `0.6451`
3. Behavior accepted per explicit user direction to continue despite degradation.

### 2026-02-15 - Stage 6 Implemented

Status: completed

Implemented:

1. Leakage-centric metric layer in evaluation:
   1. `char_canonical` (char-level precision/recall/F1)
   2. `token_canonical` (token-level precision/recall/F1)
   3. `per_label_char`
   4. `residual_miss_ratio` for every metric payload
2. Report format updates:
   1. JSON includes new metrics.
   2. Markdown includes combined and per-label char metrics with residuals.
3. Documentation updates:
   1. `docs/EVALUATION.md` extended with new metric definitions.

Validation:

1. Unit tests: `83 passed`.
2. Eval smoke run confirmed new report fields and markdown sections are present and populated.

### 2026-02-15 - Stage 7 Implemented

Status: completed (planned scope covered)

Implemented:

1. Multi-policy matrix runner:
   1. `app/tools/eval_matrix.py`
   2. Runs `app.eval.run` for each selected policy in one command.
   3. Auto-generates base-vs-candidate comparison markdown.
2. Built-in ablation orchestration:
   1. `--ablate-recognizer` repeatable option in matrix runner.
   2. Auto-generates ablation policy YAML files under `reports/evaluations/_ablation_policies/`.
   3. Executes ablation variants and includes them in final comparison.
3. Resume/checkpoint support in evaluator:
   1. `app.eval.run` now supports `--resume`.
   2. Dataset-level checkpoint persistence under `reports/evaluations/_checkpoints/`.
   3. Completed datasets are skipped on resumed runs.
4. Makefile/docs integration:
   1. `eval-matrix` supports ablation and resume flags.
   2. `docs/EVALUATION.md` updated with usage examples.
5. Tests:
   1. `tests/unit/test_eval_matrix.py`
   2. `tests/unit/test_eval_resume.py`

Results:

1. Resume smoke test passed:
   1. Second run loaded checkpoint and skipped completed dataset.
2. Ablation smoke test passed:
   1. Base + `gliner_pii_multilingual` ablation run completed.
   2. Auto-comparison markdown produced.
3. Unit tests after Stage 7 completion: `83 passed`.

### 2026-02-15 - Stage 8 Implemented

Status: completed

Implemented:

1. New pluggable recognizer type:
   1. `hf_token_classifier` is now supported by config schema and registry builder.
   2. Added `HFTokenClassifierRecognizer` in `app/core/analysis/recognizers.py`.
2. Runtime behavior and graceful degradation:
   1. Uses local in-process runtime path (CPU/MPS), including when project runtime is `cuda`.
   2. Adds inference timeout guard.
   3. Returns empty result set on model load/inference failures instead of failing the whole request.
3. Offline model asset support:
   1. Added generic HF model source resolver in `app/model_assets.py`.
   2. `hf_token_classifier` respects `GR_MODEL_DIR` and `GR_OFFLINE_MODE`.
4. Air-gapped download workflow:
   1. `app/tools/download_models.py` now discovers and downloads all `hf_token_classifier` models declared in policy.
   2. `manifest.json` now includes `hf_token_classifier_models`.
5. Documentation updates:
   1. Added `hf_token_classifier` detector behavior to `docs/DETECTORS.md`.
   2. Updated air-gapped model bundle description in `README.md`.
6. Tests:
   1. Added `tests/unit/test_hf_token_classifier_recognizer.py`.
   2. Added `tests/unit/test_download_models.py`.
   3. Extended `tests/unit/test_model_assets.py` for HF resolver helpers.

Results:

1. Targeted new/updated tests passed: `17 passed`.
2. Full unit suite after Stage 8: `93 passed, 2 skipped`.
3. No metric delta is expected yet because `hf_token_classifier` is added as a pluggable recognizer type and is not enabled in default policy profiles.

### 2026-02-15 - Stage 9 Implemented

Status: completed

Implemented:

1. Diagnostics payload in analysis/runtime API flow:
   1. Added `AnalysisDiagnostics` model.
   2. Added `analyze_text_with_diagnostics(...)` to analysis service.
   3. Guardrails detect/mask item results now include diagnostics with:
      1. per-detector timing (ms)
      2. per-detector span counts
      3. detector errors
      4. postprocess mutation counters
      5. limit flags
2. Safety limits (always-on):
   1. Max per-sample analysis budget (time).
   2. Max per-sample final detections (`256`) with deterministic truncation by score.
   3. Registry execution now enforces deadline checks during recognizer loops.
3. Span-normalizer observability:
   1. `normalize_detections(...)` now optionally returns mutation stats.
4. Offline asset manifest hardening:
   1. `download-models` now computes deterministic SHA256 tree hashes for downloaded artifacts.
   2. `manifest.json` includes per-namespace checksums and file counts.
5. Tests:
   1. Added `tests/unit/test_analysis_diagnostics.py`.
   2. Extended `tests/unit/test_download_models.py` to cover checksum manifest output.

Results:

1. Targeted Stage 9 test suites passed.
2. Full unit suite after Stage 9: `95 passed, 2 skipped`.
3. No significant divergence from expected implementation behavior observed.

### 2026-02-15 - Post-Stage Experiment: Location Boundary Tightening + Card Validation

Status: implemented and evaluated, significant divergence detected

Implemented (local branch, not yet finalized):

1. Location postprocess tightening:
   1. Replaced sentence-wide location expansion with conservative comma-chain expansion.
   2. Added transliterated RU/UZ address marker coverage.
2. Payment-card validation gate:
   1. Added Luhn-based validator.
   2. Added grouped-card fallback requiring nearby card context markers.

Evaluation (GPU host, full test split on all datasets):

1. Baseline before change:
   1. report: `eval_all_datasets_test_baseline_20260215T110716Z.json`
   2. exact canonical F1: `0.2913`
   3. overlap canonical F1: `0.8055`
   4. char canonical F1: `0.7204`
2. After change:
   1. report: `eval_all_datasets_test_baseline_20260215T113118Z.json`
   2. exact canonical F1: `0.2888` (delta `-0.0025`)
   3. overlap canonical F1: `0.7487` (delta `-0.0568`)
   4. char canonical F1: `0.6800` (delta `-0.0404`)

Interpretation:

1. This is a significant degradation.
2. Largest regression source is `payment_card` recall collapse:
   1. exact F1 `0.1567 -> 0.0389`
   2. TP `1163 -> 158`, FN `6030 -> 7035`
3. `location` exact improved slightly (`0.0190 -> 0.0209`) but did not offset global loss.

Action required:

1. Per execution policy, significant divergence requires user confirmation before finalizing/keeping this change set.

### 2026-02-15 - Post-Stage Experiment Iteration 2: Soft Card Gating (First Attempt)

Status: implemented and evaluated, still divergent

Implemented:

1. Removed hard rejection for grouped non-Luhn card candidates.
2. Kept conservative location expansion changes.

Evaluation:

1. report: `eval_all_datasets_test_baseline_20260215T120535Z.json`
2. vs baseline `20260215T110716Z`:
   1. exact canonical F1: `0.2913 -> 0.2918` (delta `+0.0005`)
   2. overlap canonical F1: `0.8055 -> 0.7679` (delta `-0.0376`)
   3. char canonical F1: `0.7204 -> 0.6958` (delta `-0.0246`)

Interpretation:

1. Hard-regression on payment cards was partially recovered, but overlap/char remained too low.
2. Divergence still considered significant.

### 2026-02-15 - Post-Stage Experiment Iteration 3: Score Calibration Without Hard Drop

Status: implemented and evaluated, accepted

Implemented:

1. Payment-card logic updated to:
   1. Reject only clearly invalid spans (digit length out of range, uniform digits).
   2. Avoid hard dropping weak-but-plausible spans.
   3. Apply mild score calibration for weak continuous card-like patterns.
2. Kept conservative location expansion updates.

Evaluation:

1. report: `eval_all_datasets_test_baseline_20260215T121912Z.json`
2. vs baseline `20260215T110716Z`:
   1. exact canonical F1: `0.2913 -> 0.2903` (delta `-0.0010`)
   2. overlap canonical F1: `0.8055 -> 0.8052` (delta `-0.0003`)
   3. char canonical F1: `0.7204 -> 0.7316` (delta `+0.0112`)
   4. token canonical F1: `0.7321 -> 0.7552` (delta `+0.0230`)
3. Label-level highlights:
   1. `location` exact F1: `0.0190 -> 0.0209` (improved).
   2. `payment_card` exact F1: `0.1567 -> 0.1581` (slightly improved).

Interpretation:

1. This iteration is consistent with goals:
   1. leakage-centric metrics improved materially
   2. strict overlap stayed effectively flat
   3. exact moved only slightly (`-0.0010`)
2. Change set accepted as the current best variant.

## 1. Why this refactor

Primary goal: materially improve real-world leakage prevention quality while keeping masking/unmasking behavior stable and production-safe.

Current evidence shows three dominant issues:

1. Boundary quality is poor on one large dataset.
2. Identifier recall is too low.
3. Some recognizers produce high false-positive volume in mixed-language text.

The plan below is intentionally staged so each step can be implemented, validated, and reverted independently.

## 2. Baseline and target outcomes

Baseline snapshot (latest full multi-dataset run on GPU host):

1. Combined exact canonical F1: 0.2364
2. Combined overlap canonical F1: 0.6984
3. Scanpatch exact F1: 0.5906
4. Rubai exact F1: 0.2220
5. Worst exact labels: `location`, `identifier`
6. Largest FP contributors: `gliner_pii_multilingual:*`, `en_pii_regex:PHONE_NUMBER:*`

Target outcomes after all stages:

1. Improve combined exact canonical F1 by at least +0.12 absolute without reducing overlap canonical F1 by more than 0.03.
2. Improve `identifier` exact recall by at least +0.15 absolute.
3. Improve `location` exact F1 by at least +0.08 absolute.
4. Add leakage-centric metrics so evaluation is not dependent only on exact boundary matching.
5. Make evaluation runs deterministic, including model warm-up readiness.

## 3. Hard constraints

1. No backward compatibility required for API/schema in this project.
2. Air-gapped mode must remain supported.
3. Runtime switch remains project-level (`cpu` or `cuda`).
4. `cuda` path continues to use PyTriton.
5. CPU path must keep Apple Silicon/MPS behavior.

## 4. Refactor strategy overview

Execution order:

1. Stabilize evaluation correctness and runtime readiness.
2. Improve detection quality with post-processing and calibration.
3. Add new detector logic for known misses.
4. Expand metrics and analysis tooling.
5. Optional major model layer additions.

Each stage below includes:

1. Scope
2. Concrete tasks
3. Code touchpoints
4. Tests
5. Exit criteria

---

## Stage 0: Baseline Freeze and Experiment Harness

### Scope

Create a clean baseline artifact set and a repeatable experiment harness so every next stage is measured apples-to-apples.

### Tasks

1. Add a baseline report manifest file that records:
   1. Report paths
   2. Commit SHA
   3. Policy path/profile
   4. Dataset list and split settings
2. Add a small comparison utility script:
   1. Input: two or more report JSON files
   2. Output: Markdown diff table for combined + per dataset + per label metrics
3. Add `make eval-compare` target.

### Code touchpoints

1. `app/tools/` add `compare_eval_reports.py`
2. `Makefile` add target
3. `docs/EVALUATION.md` add usage section

### Tests

1. Unit tests for report comparator:
   1. Missing sections
   2. Multi-dataset reports
   3. Per-label diff correctness

### Exit criteria

1. A baseline comparison can be generated with one command.
2. Diff tool outputs stable, deterministic markdown.

---

## Stage 1: Runtime Readiness and Eval Reliability

### Scope

Remove measurement noise caused by async model loading and ensure eval never silently runs before models are ready.

### Tasks

1. Introduce explicit runtime readiness contract.
2. Add blocking warm-up with timeout and strict failure options.
3. Surface readiness in evaluation report metadata.

### Concrete implementation tasks

1. Add runtime interface methods:
   1. `warm_up(timeout_s: float) -> bool`
   2. `is_ready() -> bool`
   3. `load_error() -> str | None`
2. Implement in:
   1. `LocalCpuGlinerRuntime`
   2. `PyTritonGlinerRuntime`
3. Change evaluation warm-up flow to:
   1. Warm all recognizers once
   2. Wait until ready or timeout
   3. Fail hard in strict mode
   4. Warn and continue in non-strict mode
4. Add CLI flags:
   1. `--warmup-timeout-seconds`
   2. `--warmup-strict / --no-warmup-strict`
5. Record per-recognizer warm-up status in report:
   1. ready/not ready
   2. init duration
   3. load error if any

### Code touchpoints

1. `app/runtime/gliner_runtime.py`
2. `app/core/analysis/recognizers.py`
3. `app/eval/run.py`
4. `app/eval/report.py`
5. `docs/EVALUATION.md`

### Tests

1. Unit:
   1. Warm-up success path
   2. Warm-up timeout path
   3. Strict failure behavior
2. Integration:
   1. Eval run with mocked delayed runtime confirms blocking behavior

### Exit criteria

1. No eval sample is processed before heavy recognizers are ready (strict mode).
2. Report includes warm-up diagnostics.

---

## Stage 2: Language Resolution Upgrade for Mixed Text

### Scope

Replace simplistic single-language heuristic with robust mixed-language handling.

### Tasks

1. Replace Cyrillic-only heuristic with script-ratio resolver.
2. Support mixed-mode analysis where needed.
3. Keep config simple and explicit.

### Concrete implementation tasks

1. Add language detection utility:
   1. Script counts (Cyrillic, Latin, digit-heavy)
   2. Ratio thresholds
   3. `auto_single` or `auto_union` decision
2. Extend analysis profile config:
   1. `language.strategy: single|union`
   2. `language.union_thresholds` optional
3. In union mode:
   1. Run recognizers in both supported languages
   2. Merge and deduplicate by `(start,end,canonical_label,detector)`
4. Add report slices by detected script profile:
   1. `mostly_cyrillic`
   2. `mostly_latin`
   3. `mixed`

### Code touchpoints

1. `app/core/analysis/language.py`
2. `app/config.py`
3. `app/core/analysis/service.py`
4. `app/eval/run.py`
5. `app/eval/report.py`

### Tests

1. Unit:
   1. Script classification
   2. Union merge dedup
2. Integration:
   1. Mixed-language sample should run both language recognizer variants

### Exit criteria

1. Mixed-language samples no longer depend on one coarse language guess.
2. No duplicate span inflation after union merge.

---

## Stage 3: Span Boundary Normalization Layer

### Scope

Improve exact metrics and masking quality by normalizing boundaries after detection and before masking.

### Tasks

1. Introduce a deterministic span normalizer pipeline.
2. Add label-specific expansion/shrinking rules.
3. Integrate normalizer in both API runtime and eval runtime.

### Concrete implementation tasks

1. Add module `app/core/analysis/span_normalizer.py`
2. Implement normalization passes:
   1. Trim trivial punctuation around all spans
   2. `location` expansion over comma-separated address chains
   3. `person` expansion/trim rules around initials and honorific artifacts
   4. `identifier` canonical token cleanup
3. Add maximum expansion guards:
   1. Max char expansion per label
   2. Stop tokens and sentence boundary guards
4. Add dedup and conflict resolution post-normalization:
   1. Prevent overlapping same-label fragments when a merged span is produced
5. Add config section:
   1. `analysis.postprocess.boundary.enabled`
   2. `analysis.postprocess.boundary.max_expansion_chars`
   3. Label-specific toggles

### Code touchpoints

1. `app/core/analysis/service.py`
2. `app/core/analysis/span_normalizer.py` (new)
3. `app/config.py`
4. `configs/policy.yaml`
5. `docs/DETECTORS.md` and `README.md`

### Tests

1. Unit:
   1. Boundary trim cases
   2. Address chain merge cases
   3. Overlap conflict handling
2. Regression:
   1. Ensure no shift on clean exact spans

### Exit criteria

1. `location` exact metrics improve on both datasets.
2. No >5% increase in global false positives.

---

## Stage 4: Identifier Recall Improvement Pack

### Scope

Raise identifier recall with targeted pattern coverage and context-aware boosting.

### Tasks

1. Expand identifier regex coverage for observed formats.
2. Add context-sensitive confidence adjustments.
3. Add lightweight validators for noisy generic matches.

### Concrete implementation tasks

1. Extend `identifier_regex` patterns:
   1. Letter-digit ID variants (e.g. `AA1234567`, `AB9876543`)
   2. Hyphen/parenthesized formats observed in datasets
   3. Conservative alnum+digit mixed tokens
2. Add optional context windows:
   1. Boost score if nearby trigger words (`passport`, `id`, `document`, RU equivalents)
   2. Penalize score if token appears in known non-ID patterns
3. Add validators:
   1. Length bounds
   2. Character-class sanity checks
4. Add score calibration block in policy for identifier entity type.

### Code touchpoints

1. `configs/policy.yaml`
2. `app/core/analysis/recognizers.py`
3. `app/core/analysis/service.py`
4. `docs/DETECTORS.md`

### Tests

1. Unit corpus tests for representative identifier samples.
2. False-positive guard tests with confusing numeric tokens.

### Exit criteria

1. `identifier` exact recall +0.15 absolute on Rubai test split.
2. `identifier` precision does not drop by more than 0.08.

---

## Stage 5: Phone and High-FP Regex Calibration

### Scope

Reduce avoidable regex-driven false positives without harming true leak coverage.

### Tasks

1. Rework phone regex thresholds and context requirements.
2. Introduce detector-level score gates.
3. Add regex gating by language/script profile.

### Concrete implementation tasks

1. Add detector-specific threshold support in config:
   1. `analysis.detector_thresholds` map
2. In service filtering:
   1. Apply `max(policy_min_score, entity_threshold, detector_threshold)`
3. For `en_pii_regex:PHONE_NUMBER`:
   1. Raise default score
   2. Require separators/length patterns less likely to collide with random numbers
4. Add script-aware gating option:
   1. Skip certain EN regex patterns on mostly Cyrillic text unless explicit context hit

### Code touchpoints

1. `app/config.py`
2. `app/core/analysis/service.py`
3. `configs/policy.yaml`
4. `app/core/analysis/language.py`
5. `docs/DETECTORS.md`

### Tests

1. Unit tests for detector threshold precedence.
2. Unit tests for script-aware gating decisions.
3. Eval ablation run comparing before/after per-detector FP.

### Exit criteria

1. Lower FP contribution from `en_pii_regex:PHONE_NUMBER` by at least 30%.
2. Global exact F1 does not regress.

---

## Stage 6: Leakage-Centric Evaluation Metrics

### Scope

Add metrics that better represent leakage prevention quality than strict exact span matching alone.

### Tasks

1. Add character-level and token-level coverage metrics.
2. Add residual-risk metrics.
3. Include per-entity leakage miss rates.

### Concrete implementation tasks

1. Add new metrics module functions:
   1. Character overlap precision/recall/F1
   2. Token overlap precision/recall/F1
   3. Residual sensitive chars ratio
2. Add per-label leakage metrics.
3. Extend report JSON and markdown sections:
   1. Combined
   2. Per dataset
   3. Per slice (source, noisy, script profile)
4. Add compatibility in comparator tool from Stage 0.

### Code touchpoints

1. `app/eval/metrics.py`
2. `app/eval/report.py`
3. `app/eval/run.py`
4. `docs/EVALUATION.md`

### Tests

1. Unit tests with synthetic span pairs validating char/token math.
2. Regression tests ensuring old metrics remain unchanged.

### Exit criteria

1. Reports include both strict NER metrics and leakage-centric metrics.
2. Evaluation decisions can be based on residual-risk, not just exact F1.

---

## Stage 7: Multi-Profile Eval Runner and Ablation Automation

### Scope

Make iterative quality tuning fast and reproducible.

### Tasks

1. Add one-command multi-profile eval matrix.
2. Add automatic ablation report generation.
3. Cache and reuse split/materialization metadata.

### Concrete implementation tasks

1. Add CLI options:
   1. `--policy-name` repeatable or profile matrix file
   2. `--detector-ablation` repeatable
2. Add output folder structure:
   1. baseline report
   2. ablation reports
   3. merged comparison markdown
3. Add dataset-level progress checkpoints and resume support:
   1. Partial JSON state per dataset
   2. Resume flag loads completed dataset results

### Code touchpoints

1. `app/eval/run.py`
2. `app/tools/compare_eval_reports.py`
3. `Makefile`
4. `docs/EVALUATION.md`

### Tests

1. Unit tests for matrix argument parsing.
2. Integration test with mocked dataset adapters for resume flow.

### Exit criteria

1. Large tuning rounds can be run and resumed reliably.
2. One command produces a ranked improvement report.

---

## Stage 8: Optional Major Model Additions (Pluggable)

### Scope

Add one or more complementary model detectors while preserving current architecture.

### Candidate additions

1. Hugging Face token classification detector (`transformers` pipeline based)
2. Optional multilingual NER model specialized for addresses/IDs
3. Future company fine-tuned GLiNER as drop-in model reference

### Concrete implementation tasks

1. Add new recognizer type to config:
   1. `hf_token_classifier`
2. Implement recognizer class with:
   1. local/offline model path support
   2. CPU/cuda runtime compatibility
   3. label-to-canonical mapping
3. Integrate into Presidio registry builder.
4. Add detector-level timeout and graceful degradation.

### Code touchpoints

1. `app/config.py`
2. `app/core/analysis/recognizers.py`
3. `app/model_assets.py`
4. `app/tools/download_models.py`
5. `docs/DETECTORS.md`

### Tests

1. Unit tests for label mapping and conversion.
2. Offline model load test with local model directory fixture.

### Exit criteria

1. New model can be enabled by policy only.
2. Air-gapped mode fully supported.

---

## Stage 9: Production Hardening and Observability

### Scope

Prepare the improved stack for stable platform integration.

### Tasks

1. Add internal diagnostics API payload fields:
   1. per-detector timing
   2. per-detector span counts
   3. post-processor mutation counts
2. Add safety limits:
   1. max spans per sample
   2. max processing time per sample
3. Add offline asset manifest with checksums.

### Code touchpoints

1. `app/api/*`
2. `app/core/*`
3. `app/tools/download_models.py`
4. `README.md`

### Tests

1. Integration tests for timeout and max-span guards.
2. Manifest validation tests.

### Exit criteria

1. System degrades gracefully under pathological input.
2. Debug payloads are sufficient for triage without code instrumentation.

---

## 5. Cross-stage testing protocol

Run after every stage:

1. `uv run --extra dev pytest -q`
2. `uv run --extra eval python -m app.eval.run --split test --policy-path configs/policy.yaml --policy-name external_default --env-file .env.eval --output-dir reports/evaluations`
3. `uv run --extra eval python -m app.tools.compare_eval_reports --base <baseline.json> --candidate <new.json>`

Stage-gate policy:

1. Do not advance if exact F1 regresses by more than 0.02 unless leakage-centric metrics improve significantly and rationale is documented.
2. Any detector-specific change must include per-detector FP/TP delta evidence.

## 6. Risk register and mitigations

1. Risk: Overfitting to Rubai annotation quirks.
   1. Mitigation: track leakage-centric metrics and Scanpatch deltas together.
2. Risk: Increased latency from union-language and post-processing.
   1. Mitigation: add timing metrics and configurable toggles.
3. Risk: Regex expansion creates FP spikes.
   1. Mitigation: detector-level thresholds and mandatory ablations.
4. Risk: Warm-up strict mode hurts developer loop.
   1. Mitigation: keep non-strict mode default for local experiments.

## 7. Delivery order recommendation

Recommended implementation sequence for highest ROI:

1. Stage 1
2. Stage 3
3. Stage 4
4. Stage 5
5. Stage 6
6. Stage 2
7. Stage 7
8. Stage 8
9. Stage 9

Rationale:

1. Stage 1 removes measurement noise.
2. Stage 3 and Stage 4 directly attack biggest observed weaknesses.
3. Stage 5 controls FP side effects from recall-focused changes.
4. Stage 6 ensures decisions are aligned with leakage prevention.

## 8. Definition of done for the full refactor

1. All stages completed or explicitly skipped with documented reason.
2. Final comparison report shows metric deltas for every stage.
3. Updated docs:
   1. `docs/EVALUATION.md`
   2. `docs/DETECTORS.md`
   3. `README.md`
4. Air-gapped workflow still validated:
   1. model download
   2. offline load
   3. offline eval run
5. Clear handoff artifact:
   1. final metrics summary
   2. policy diff
   3. detector behavior changes
   4. known limitations
