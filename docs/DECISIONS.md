# Architecture Decisions

This file records non-trivial guardrails decisions that changed default behavior.

## 2026-02-16: Roll Back Nemotron Person Labels in Default Policy

Status:
- Accepted and implemented.

Context:
- Baseline reference was captured in `baselines/BASELINE_1` (commit `a1215e28a32074fbb6c8b8637c8b0ecad8425e72`).
- Full test-split comparison showed a major degradation after recent NER changes:
  - Combined exact F1: `0.308670 -> 0.226051`
  - Combined overlap F1: `0.851318 -> 0.733781`
  - Combined person exact F1: `0.451914 -> 0.106946`
- Most regression came from `BoburAmirov/rubai-NER-150K-Personal`; `scanpatch` stayed roughly stable.

Evidence:
- Ablation on rubai test subset (6000 samples):
  - Control: person F1 `0.106370`
  - No Natasha: person F1 `0.106042` (no meaningful change)
  - No Nemotron person labels: person F1 `0.513467` (large recovery)
  - No Natasha + no Nemotron person labels: person F1 `0.513930`
- Synthetic RU story in `.local/synthetic_cases/ru_story_case_001.txt` still detects core entities correctly with GLiNER + regex + structured Nemotron labels.

Decision:
1. Remove Natasha from default `external_rich` execution path.
2. Keep Natasha recognizer definition in policy, but disable it by default.
3. Keep Nemotron enabled as an optional detector runtime, but remove person-name raw labels from default policy:
   - removed: `name`, `first_name`, `last_name`, `middle_name`, `name_initials`, `nickname`
4. Keep Nemotron focused on structured PII labels in default policy.
5. Add evaluator regression gate for full test-split runs on rubai:
   - minimum `person` exact precision: `0.30`
   - minimum `person` exact F1: `0.35`
   - gate is skipped for sampled runs (`--max-samples`).

Consequences:
- Better default precision/quality stability for person detection on rubai-like data.
- Fewer surprise regressions in manual benchmark runs due to hard gate failure.
- Natasha remains available for future controlled experiments, not active by default.
