# Detector Catalog

This document maps policy recognizers to concrete behavior, output entity labels, and examples.

Maintenance rule:
- If you change `configs/policy.yaml` recognizer definitions or recognizer code in `app/core/analysis/recognizers.py`, update this file in the same commit.

## Output model

Runtime execution model:
- The detector pipeline is language-agnostic at runtime.
- All configured recognizers run in a single pass per text (no language detection and no per-language routing).

Each finding has:
- `entity_type`: Presidio-style entity (for example `EMAIL_ADDRESS`, `PHONE_NUMBER`, `DOCUMENT_NUMBER`).
- `label`: service label used for masking (currently canonical label uppercased, for example `EMAIL`, `PHONE`, `IDENTIFIER`, `SECRET`).
- `canonical_label`: normalized taxonomy used by policy/reporting (`email`, `phone`, `identifier`, `secret`, etc).

## Recognizer: `phone_number_lib`

Type: `phone`

Source:
- `phonenumbers` validation and parsing.

Primary entity labels:
- `PHONE_NUMBER`

Examples:
- `+7 (999) 123-45-67` -> `PHONE_NUMBER`
- `+1 (415) 555-0123` -> `PHONE_NUMBER`

## Recognizer: `ip_address_lib`

Type: `ip`

Source:
- Python `ipaddress` parsing with IPv4/IPv6/CIDR support.

Primary entity labels:
- `IP_ADDRESS`

Examples:
- `2001:DB8::FFFF:10.10.2.1` -> `IP_ADDRESS`
- `10.12.0.0/16` -> `IP_ADDRESS`

## Recognizer: `ru_pii_regex`

Type: `regex`

Pattern labels:
- `PHONE_NUMBER`
- `DOCUMENT_NUMBER`
- `TIN`
- `CREDIT_CARD`
- `EMAIL_ADDRESS`

Examples:
- `Паспорт: 1234 567890` -> `DOCUMENT_NUMBER`
- `ИНН 7707083893` -> `TIN`
- `ivan.petrov@example.com` -> `EMAIL_ADDRESS`

## Recognizer: `en_pii_regex`

Type: `regex`

Pattern labels:
- `US_SSN`
- `IBAN_CODE`
- `SWIFT_CODE`

Examples:
- `SSN: 123-45-6789` -> `US_SSN`
- `IBAN DE89370400440532013000` -> `IBAN_CODE`
- `BIC DEUTDEFF` -> `SWIFT_CODE`

## Recognizer: `identifier_regex`

Type: `regex`

Pattern labels:
- `MILITARY_INDIVIDUAL_NUMBER`
- `VEHICLE_NUMBER`
- `DOCUMENT_NUMBER`

Examples:
- `в/ч-5211` -> `MILITARY_INDIVIDUAL_NUMBER`
- `КА9914МВ` -> `VEHICLE_NUMBER`
- `UPZ-11903` -> `DOCUMENT_NUMBER`

## Recognizer: `network_pii_regex`

Type: `regex`

Pattern labels:
- `IP_ADDRESS`

Examples:
- `193.51.208.14` -> `IP_ADDRESS`
- `100.64.0.0/10` -> `IP_ADDRESS`

## Recognizer: `date_pii_regex`

Type: `regex`

Pattern labels:
- `DATE_TIME`

Examples:
- `12.03.2022` -> `DATE_TIME`
- `2024.Q3` -> `DATE_TIME`

## Recognizer: `code_secret_regex`

Type: `secret_regex`

Pattern labels:
- `API_KEY`

Concrete pattern intents:
- AWS access key
- GitHub token
- Slack token
- generic key assignment
- private key PEM header
- JWT token

Examples:
- `AWS_ACCESS_KEY_EXAMPLE_0001` -> `API_KEY`
- `github_token_example_value_0001` -> `API_KEY`
- `-----BEGIN RSA PRIVATE KEY-----` -> `API_KEY`

## Recognizer: `high_entropy_secret`

Type: `entropy`

Entity labels:
- `API_KEY`

Heuristic:
- regex token candidate + minimum length + Shannon entropy threshold.

Examples:
- `stripe_example_key_value_0001` -> `API_KEY`

## Recognizer: `gliner_pii_multilingual`

Type: `gliner`

Source:
- GLiNER model (`urchade/gliner_multi-v2.1`) via local runtime (`cpu`) or PyTriton (`cuda`).

Policy notes:
- Input is chunked automatically based on the model context length using tokenizer offsets (full coverage, no chunk caps).

Configured query labels:
- `person`
- `organization`
- `location`
- `address`
- `street address`
- `city`
- `district`
- `postal code`
- `document number`
- `tax identification number`
- `snils`
- `military number`
- `vehicle number`
- `email`
- `phone number`
- `ip address`
- `date`
- `passport number`
- `credit card number`
- `api key`

Normalized entity labels produced by recognizer:
- `PERSON`
- `ORGANIZATION`
- `LOCATION`
- `DOCUMENT_NUMBER`
- `EMAIL_ADDRESS`
- `PHONE_NUMBER`
- `IP_ADDRESS`
- `DATE_TIME`
- `CREDIT_CARD`
- `API_KEY`

Examples:
- `Контакт: ivan@example.com` -> `EMAIL_ADDRESS`
- `Телефон +7 999 123 45 67` -> `PHONE_NUMBER`

## Recognizer: `natasha_ner_ru`

Type: `natasha_ner`

Source:
- Natasha NER stack (`Segmenter`, `NewsEmbedding`, `NewsNERTagger`).

Raw model labels:
- `PER`
- `ORG`
- `LOC`

Configured mapping to normalized entities:
- `PERSON`
- `ORGANIZATION`
- `LOCATION`

Policy notes:
- Recognizer is disabled by default in `configs/policy.yaml` (kept as an optional detector for targeted experiments).
- Imperative prompt verbs at sentence start are filtered for `PER` (for example, `Перескажи` false positives).

Examples:
- `Ваня Миллипиздриков встретил Григория Стрельникова` -> `PERSON`
- `Компания Ромашка` -> `ORGANIZATION`
- `Москва` -> `LOCATION`

## Recognizer: `nemotron_pii_token_classifier`

Type: `token_classifier`

Source:
- Hugging Face token-classification model `scanpatch/pii-ner-nemotron` (`transformers`, XLM-RoBERTa family).
- Runtime:
- `cpu` mode: local transformers inference (CPU/MPS auto-selection).
- `cuda` mode: PyTriton-hosted model (served as `nemotron`).

Raw model labels (BIO collapsed to entity group):
- `address`, `address_apartment`, `address_building`, `address_city`, `address_country`, `address_district`, `address_geolocation`, `address_house`, `address_postal_code`, `address_region`, `address_street`
- `date`
- `document_number`
- `email`
- `ip`
- `military_individual_number`
- `mobile_phone`
- `snils`
- `tin`
- `vehicle_number`

Configured mapping to normalized entities:
- `LOCATION`
- `DATE_TIME`
- `DOCUMENT_NUMBER`
- `EMAIL_ADDRESS`
- `IP_ADDRESS`
- `PHONE_NUMBER`
- `TIN`

Policy notes:
- Recognizer is loaded only when `GR_ENABLE_NEMOTRON=true` (default is disabled).
- Default runtime policy keeps Nemotron on structured PII labels and intentionally excludes person-name labels to prevent large false-positive spikes on public benchmarks.
- Per-entity minimum confidence thresholds are applied in recognizer postprocessing (`entity_thresholds` / `raw_label_thresholds`).
- Input is chunked automatically based on the model context length using tokenizer offsets (full coverage, no chunk caps).

Examples:
- `Проживает: г. Казань, ул. Пушкина 12` -> `LOCATION`
- `Почта: ivan.petrov@corp.local` -> `EMAIL_ADDRESS`
- `ИНН 7707083893` -> `TIN`
