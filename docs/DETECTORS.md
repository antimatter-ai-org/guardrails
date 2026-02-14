# Detector Catalog

This document is the operator-facing reference for detector behavior in this repository.

Maintenance rule:
- If you add, remove, rename, or relabel a detector in `configs/policy.yaml` or detector code, update this file in the same change.

## Label format

- `regex` and `secret_regex` detectors emit labels exactly as configured in pattern definitions.
- `entropy` detector emits the constant label `SECRET_HIGH_ENTROPY`.
- `natasha` detector emits `NER_{TYPE}` (for example `NER_PER`).
- `gliner` detector emits `GLINER_{label_from_model}` (for example `GLINER_person`).

## Detector: `ru_pii_regex`

Type: `regex`

Purpose:
- Detect Russian-centric PII and common identifiers.

Concrete labels used:
- `RU_PHONE`
- `RU_PASSPORT`
- `RU_SNILS`
- `RU_INN`
- `RU_OGRN`
- `PAYMENT_CARD`
- `EMAIL`

Examples:
- Text: `Свяжитесь со мной: +7 (999) 123-45-67` -> label `RU_PHONE`
- Text: `Паспорт: 1234 567890` -> label `RU_PASSPORT`
- Text: `SNILS 123-456-789 01` -> label `RU_SNILS`
- Text: `ИНН 7707083893` -> label `RU_INN`
- Text: `ОГРН 1027700132195` -> label `RU_OGRN`
- Text: `Карта 4111 1111 1111 1111` -> label `PAYMENT_CARD`
- Text: `ivan.petrov@example.com` -> label `EMAIL`

## Detector: `en_pii_regex`

Type: `regex`

Purpose:
- Detect English/international PII and payment/account identifiers.

Concrete labels used:
- `US_SSN`
- `IBAN`
- `SWIFT`
- `PHONE`

Examples:
- Text: `SSN: 123-45-6789` -> label `US_SSN`
- Text: `IBAN DE89370400440532013000` -> label `IBAN`
- Text: `BIC DEUTDEFF` -> label `SWIFT`
- Text: `Call me at +1 (415) 555-0123` -> label `PHONE`

## Detector: `code_secret_regex`

Type: `secret_regex`

Purpose:
- Detect secrets commonly found in source code, logs, and agent tool output.

Concrete labels used (built-in):
- `SECRET_AWS_ACCESS_KEY`
- `SECRET_GITHUB_TOKEN`
- `SECRET_SLACK_TOKEN`
- `SECRET_GENERIC_KEY`
- `SECRET_PRIVATE_KEY`

Concrete labels used (configured extension):
- `SECRET_JWT`

Examples:
- Text: `AWS_ACCESS_KEY_EXAMPLE_0001` -> label `SECRET_AWS_ACCESS_KEY`
- Text: `github_token_example_value_0002` -> label `SECRET_GITHUB_TOKEN`
- Text: `slack_token_example_value_0001` -> label `SECRET_SLACK_TOKEN`
- Text: `api_key = "stripe_example_key_value_0003"` -> label `SECRET_GENERIC_KEY`
- Text: `-----BEGIN RSA PRIVATE KEY-----` -> label `SECRET_PRIVATE_KEY`
- Text: `eyJhbGciOi...<snip>...signature` -> label `SECRET_JWT`

## Detector: `high_entropy_secret`

Type: `entropy`

Purpose:
- Catch unknown secret formats via high-entropy token detection.

Concrete labels used:
- `SECRET_HIGH_ENTROPY`

Examples:
- Text: `stripe_example_key_value_0001` -> label `SECRET_HIGH_ENTROPY`
- Text: `3M6f9xQz7Wv2sK1n8Tg5Yp4Rj0LcH2` -> label `SECRET_HIGH_ENTROPY`

## Detector: `natasha_ner_ru`

Type: `natasha`

Purpose:
- Russian NER for person, organization, and location entities.

Concrete labels used:
- `NER_PER`
- `NER_ORG`
- `NER_LOC`

Examples:
- Text: `Иван Петров работает в Сбере` -> labels `NER_PER`, `NER_ORG`
- Text: `Офис находится в Москве` -> label `NER_LOC`

Notes:
- This detector uses Natasha `NewsNERTagger` (which is backed by Slovnet models internally).

## Detector: `gliner_pii_multilingual`

Type: `gliner`

Default status:
- Configured but disabled (`enabled: false`) in `configs/policy.yaml`.

Purpose:
- Optional multilingual transformer-based entity detector for additional recall.

Configured candidate labels:
- `person`
- `organization`
- `location`
- `email`
- `phone number`
- `passport number`
- `credit card number`
- `api key`

Concrete emitted labels (with current prefixing rule):
- `GLINER_person`
- `GLINER_organization`
- `GLINER_location`
- `GLINER_email`
- `GLINER_phone number`
- `GLINER_passport number`
- `GLINER_credit card number`
- `GLINER_api key`

Examples:
- Text: `Contact: ivan@example.com` -> likely `GLINER_email`
- Text: `Телефон +7 999 123 45 67` -> likely `GLINER_phone number`
