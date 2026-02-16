# Data Assets Tracker

Tracks potentially useful external assets for the Guardrails project.

Scope:
- Primary languages: Russian (`ru`) and English (`en`)
- MVP focus: PII and sensitive data masking (including secrets/host/network-related identifiers)
- Format: candidate assets to review, test, and optionally incorporate into eval/training workflows

## Models

| Asset | Type | Language Coverage | Status | Note |
|---|---|---|---|---|
| `tabularisai/eu-pii-safeguard` | Token-classification model | `ru`, `en`, plus other EU languages | Candidate (baseline model) | This is a **model repo**, not a dataset repo. Useful as a baseline detector benchmark against our derived eval datasets after label mapping to our canonical schema. |
