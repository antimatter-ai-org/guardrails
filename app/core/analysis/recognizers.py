from __future__ import annotations

import ipaddress
import math
from typing import Any

import regex as re
from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerRegistry, RecognizerResult

from app.config import RecognizerDefinition
from app.runtime.gliner_runtime import build_gliner_runtime
from app.runtime.token_classifier_runtime import build_token_classifier_runtime
from app.settings import settings

_FLAG_MAP: dict[str, int] = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "UNICODE": re.UNICODE,
}
_PRESIDIO_COMPAT_LANGUAGE = "global"
_NATASHA_PROMPT_IMPERATIVE_WORDS = {
    "перескажи",
    "расскажи",
    "напиши",
    "составь",
    "придумай",
    "объясни",
    "переведи",
    "отредактируй",
    "сократи",
    "продолжи",
    "перефразируй",
}


def _normalize_entity_type(label: str) -> str:
    normalized = label.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized in {"PERSON", "FULL_NAME", "FIRST_NAME", "MIDDLE_NAME", "LAST_NAME", "NAME"}:
        return "PERSON"
    if normalized in {"ORGANIZATION", "ORG"}:
        return "ORGANIZATION"
    if normalized in {"LOCATION", "ADDRESS", "CITY", "DISTRICT", "POSTAL_CODE", "STREET_ADDRESS"}:
        return "LOCATION"
    if normalized.startswith("ADDRESS_"):
        return "LOCATION"
    if normalized in {"EMAIL", "EMAIL_ADDRESS"}:
        return "EMAIL_ADDRESS"
    if normalized in {"PHONE", "PHONE_NUMBER", "MOBILE_PHONE"}:
        return "PHONE_NUMBER"
    if normalized in {"IP", "IP_ADDRESS"}:
        return "IP_ADDRESS"
    if normalized in {"DATE", "DATE_TIME"}:
        return "DATE_TIME"
    if normalized in {"PASSPORT_NUMBER", "DOCUMENT_NUMBER", "SNILS", "TAX_IDENTIFICATION_NUMBER", "TIN"}:
        return "DOCUMENT_NUMBER"
    if normalized in {"MILITARY_NUMBER", "MILITARY_INDIVIDUAL_NUMBER", "VEHICLE_NUMBER"}:
        return "DOCUMENT_NUMBER"
    if normalized in {"CREDIT_CARD_NUMBER", "CREDIT_CARD", "PAYMENT_CARD"}:
        return "CREDIT_CARD"
    if normalized in {"API_KEY", "SECRET", "SECRET_KEY"}:
        return "API_KEY"
    return normalized


class PhoneNumberRecognizer(EntityRecognizer):
    def __init__(
        self,
        *,
        name: str,
        supported_language: str,
        score: float,
        regions: list[str],
        min_digits: int,
    ) -> None:
        self._score = float(score)
        self._regions = [item.strip().upper() for item in regions if item.strip()]
        self._min_digits = max(6, int(min_digits))
        super().__init__(
            supported_entities=["PHONE_NUMBER"],
            name=name,
            supported_language=supported_language,
        )

    def load(self) -> None:
        from phonenumbers import Leniency, PhoneNumberMatcher, is_valid_number

        self._matcher_cls = PhoneNumberMatcher
        self._leniency = Leniency
        self._is_valid_number = is_valid_number

    def analyze(self, text: str, entities: list[str], nlp_artifacts: Any = None) -> list[RecognizerResult]:
        if entities and "PHONE_NUMBER" not in entities:
            return []

        hits: list[RecognizerResult] = []
        seen: set[tuple[int, int]] = set()
        for region in self._regions:
            matcher = self._matcher_cls(text, region, leniency=self._leniency.VALID)
            for match in matcher:
                start = int(match.start)
                end = int(match.end)
                if end <= start or (start, end) in seen:
                    continue
                raw = text[start:end]
                if sum(char.isdigit() for char in raw) < self._min_digits:
                    continue
                if not self._is_valid_number(match.number):
                    continue
                seen.add((start, end))
                hits.append(
                    RecognizerResult(
                        entity_type="PHONE_NUMBER",
                        start=start,
                        end=end,
                        score=self._score,
                        recognition_metadata={
                            RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                            RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                        },
                    )
                )
        return hits


class IPAddressRecognizer(EntityRecognizer):
    def __init__(self, *, name: str, supported_language: str, score: float) -> None:
        self._score = float(score)
        super().__init__(
            supported_entities=["IP_ADDRESS"],
            name=name,
            supported_language=supported_language,
        )

    def load(self) -> None:
        self._candidate_re = re.compile(r"(?<![A-Za-z0-9_])(?:[0-9A-Fa-f:.]{2,}(?:/[0-9]{1,3})?)(?![A-Za-z0-9_])")

    @staticmethod
    def _trim_token(raw: str) -> tuple[str, int, int]:
        trim_chars = "[](){}<>,;\"'`"
        left = 0
        right = len(raw)
        while left < right and raw[left] in trim_chars:
            left += 1
        while right > left and raw[right - 1] in trim_chars:
            right -= 1
        return raw[left:right], left, len(raw) - right

    @staticmethod
    def _is_ip(token: str) -> bool:
        separators = token.count(".") + token.count(":")
        if separators < 2:
            return False
        try:
            if "/" in token:
                ipaddress.ip_network(token, strict=False)
            else:
                ipaddress.ip_address(token)
        except Exception:
            return False
        return True

    def analyze(self, text: str, entities: list[str], nlp_artifacts: Any = None) -> list[RecognizerResult]:
        if entities and "IP_ADDRESS" not in entities:
            return []
        results: list[RecognizerResult] = []
        seen: set[tuple[int, int]] = set()
        for match in self._candidate_re.finditer(text):
            raw = match.group(0)
            token, left_trim, right_trim = self._trim_token(raw)
            if not token or not self._is_ip(token):
                continue
            start = match.start() + left_trim
            end = match.end() - right_trim
            if end <= start or (start, end) in seen:
                continue
            seen.add((start, end))
            results.append(
                RecognizerResult(
                    entity_type="IP_ADDRESS",
                    start=start,
                    end=end,
                    score=self._score,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                    },
                )
            )
        return results


class GlinerPresidioRecognizer(EntityRecognizer):
    def __init__(
        self,
        *,
        name: str,
        supported_language: str,
        model_name: str,
        labels: list[str],
        threshold: float,
        triton_model_name: str = "gliner",
        chunking: dict[str, Any] | None = None,
    ) -> None:
        self._labels = [item for item in labels if item]
        self._threshold = float(threshold)
        self._runtime = build_gliner_runtime(
            runtime_mode=settings.runtime_mode,
            model_name=model_name,
            cpu_device=settings.cpu_device,
            pytriton_url=settings.pytriton_url,
            pytriton_model_name=triton_model_name,
            pytriton_init_timeout_s=settings.pytriton_init_timeout_s,
            pytriton_infer_timeout_s=settings.pytriton_infer_timeout_s,
            chunking_enabled=bool((chunking or {}).get("enabled", True)),
            chunking_max_tokens=int((chunking or {}).get("max_tokens", 320)),
            chunking_overlap_tokens=int((chunking or {}).get("overlap_tokens", 64)),
            chunking_max_chunks=int((chunking or {}).get("max_chunks", 64)),
            chunking_boundary_lookback_tokens=int((chunking or {}).get("boundary_lookback_tokens", 24)),
        )
        entities = sorted({_normalize_entity_type(label) for label in self._labels})
        super().__init__(
            supported_entities=entities,
            name=name,
            supported_language=supported_language,
        )

    def load(self) -> None:
        return None

    def analyze(self, text: str, entities: list[str], nlp_artifacts: Any = None) -> list[RecognizerResult]:
        raw_predictions = self._runtime.predict_entities(text, self._labels, threshold=self._threshold)
        results: list[RecognizerResult] = []
        for item in raw_predictions:
            label = str(item.get("label", ""))
            entity_type = _normalize_entity_type(label)
            if entities and entity_type not in entities:
                continue
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            if end <= start:
                continue
            score = float(item.get("score", self._threshold))
            results.append(
                RecognizerResult(
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    score=score,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                    },
                )
            )
        return results


class TokenClassifierPresidioRecognizer(EntityRecognizer):
    def __init__(
        self,
        *,
        name: str,
        supported_language: str,
        model_name: str,
        threshold: float,
        labels: list[str],
        label_mapping: dict[str, str] | None = None,
        raw_label_thresholds: dict[str, float] | None = None,
        entity_thresholds: dict[str, float] | None = None,
        aggregation_strategy: str = "simple",
        triton_model_name: str = "nemotron",
        chunking: dict[str, Any] | None = None,
    ) -> None:
        self._threshold = float(threshold)
        self._labels = [str(item).strip() for item in labels if str(item).strip()]
        self._label_mapping = {
            str(key).strip().lower(): _normalize_entity_type(str(value))
            for key, value in (label_mapping or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self._raw_label_thresholds = {
            str(key).strip().lower(): float(value)
            for key, value in (raw_label_thresholds or {}).items()
            if str(key).strip()
        }
        self._entity_thresholds = {
            _normalize_entity_type(str(key)): float(value)
            for key, value in (entity_thresholds or {}).items()
            if str(key).strip()
        }
        self._runtime = build_token_classifier_runtime(
            runtime_mode=settings.runtime_mode,
            model_name=model_name,
            cpu_device=settings.cpu_device,
            pytriton_url=settings.pytriton_url,
            pytriton_model_name=triton_model_name,
            pytriton_init_timeout_s=settings.pytriton_init_timeout_s,
            pytriton_infer_timeout_s=settings.pytriton_infer_timeout_s,
            aggregation_strategy=aggregation_strategy,
            chunking_enabled=bool((chunking or {}).get("enabled", True)),
            chunking_max_tokens=int((chunking or {}).get("max_tokens", 320)),
            chunking_overlap_tokens=int((chunking or {}).get("overlap_tokens", 64)),
            chunking_max_chunks=int((chunking or {}).get("max_chunks", 64)),
            chunking_boundary_lookback_tokens=int((chunking or {}).get("boundary_lookback_tokens", 24)),
        )
        # Keep entities broad; exact labels are normalized per model output.
        entities = [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "IP_ADDRESS",
            "DATE_TIME",
            "DOCUMENT_NUMBER",
            "TIN",
            "CREDIT_CARD",
            "API_KEY",
        ]
        super().__init__(
            supported_entities=entities,
            name=name,
            supported_language=supported_language,
        )

    def load(self) -> None:
        return None

    def _map_label(self, label: str) -> str:
        normalized = label.strip().lower()
        if normalized.startswith("b-") or normalized.startswith("i-"):
            normalized = normalized[2:]
        mapped = self._label_mapping.get(normalized)
        if mapped:
            return mapped
        return _normalize_entity_type(label)

    def analyze(self, text: str, entities: list[str], nlp_artifacts: Any = None) -> list[RecognizerResult]:
        raw_predictions = self._runtime.predict_entities(text, self._labels, threshold=self._threshold)
        results: list[RecognizerResult] = []
        for item in raw_predictions:
            label = str(item.get("label", ""))
            if not label:
                continue
            normalized_raw = label.strip().lower()
            if normalized_raw.startswith("b-") or normalized_raw.startswith("i-"):
                normalized_raw = normalized_raw[2:]
            entity_type = self._map_label(label)
            if entities and entity_type not in entities:
                continue
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            if end <= start:
                continue
            score = float(item.get("score", self._threshold))
            min_score = self._entity_thresholds.get(
                entity_type,
                self._raw_label_thresholds.get(normalized_raw, self._threshold),
            )
            if score < min_score:
                continue
            results.append(
                RecognizerResult(
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    score=score,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                    },
                )
            )
        return results


class NatashaNerRecognizer(EntityRecognizer):
    def __init__(
        self,
        *,
        name: str,
        supported_language: str,
        score: float,
        drop_prompt_imperatives: bool = True,
        min_person_chars: int = 3,
    ) -> None:
        self._score = float(score)
        self._drop_prompt_imperatives = bool(drop_prompt_imperatives)
        self._min_person_chars = max(2, int(min_person_chars))
        self._load_error: str | None = None
        self._segmenter: Any | None = None
        self._tagger: Any | None = None
        self._doc_cls: Any | None = None
        super().__init__(
            supported_entities=["PERSON", "ORGANIZATION", "LOCATION"],
            name=name,
            supported_language=supported_language,
        )

    def load(self) -> None:
        if self._segmenter is not None and self._tagger is not None and self._doc_cls is not None:
            return
        try:
            from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
        except Exception as exc:
            self._load_error = f"natasha import error: {exc}"
            return
        try:
            self._segmenter = Segmenter()
            embedding = NewsEmbedding()
            self._tagger = NewsNERTagger(embedding)
            self._doc_cls = Doc
            self._load_error = None
        except Exception as exc:
            self._load_error = f"natasha runtime init error: {exc}"
            self._segmenter = None
            self._tagger = None
            self._doc_cls = None

    @staticmethod
    def _is_sentence_start(text: str, start: int) -> bool:
        cursor = max(0, int(start)) - 1
        while cursor >= 0 and text[cursor].isspace():
            cursor -= 1
        if cursor < 0:
            return True
        return text[cursor] in ".!?\n\r:;"

    def _drop_prompt_person_false_positive(self, *, text: str, start: int, end: int) -> bool:
        segment = text[start:end].strip()
        if not segment:
            return True
        words = re.findall(r"\p{L}+", segment)
        if not words:
            return True
        compact_len = sum(len(word) for word in words)
        if compact_len < self._min_person_chars:
            return True
        if len(words) != 1 or not self._drop_prompt_imperatives:
            return False
        if not self._is_sentence_start(text, start):
            return False
        return words[0].lower() in _NATASHA_PROMPT_IMPERATIVE_WORDS

    def analyze(self, text: str, entities: list[str], nlp_artifacts: Any = None) -> list[RecognizerResult]:
        if entities and not set(entities).intersection({"PERSON", "ORGANIZATION", "LOCATION"}):
            return []
        if self._segmenter is None or self._tagger is None or self._doc_cls is None:
            self.load()
        if self._segmenter is None or self._tagger is None or self._doc_cls is None:
            return []

        results: list[RecognizerResult] = []
        mapping = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION"}

        try:
            doc = self._doc_cls(text)
            doc.segment(self._segmenter)
            doc.tag_ner(self._tagger)
            spans = list(getattr(doc, "spans", []) or [])
        except Exception:
            return []

        for span in spans:
            entity_type = mapping.get(str(getattr(span, "type", "")).upper())
            if entity_type is None:
                continue
            if entities and entity_type not in entities:
                continue
            start = int(getattr(span, "start", -1))
            end = int(getattr(span, "stop", -1))
            if end <= start:
                continue
            if entity_type == "PERSON" and self._drop_prompt_person_false_positive(text=text, start=start, end=end):
                continue
            results.append(
                RecognizerResult(
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    score=self._score,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                    },
                )
            )
        return results


class EntropySecretRecognizer(EntityRecognizer):
    def __init__(
        self,
        *,
        name: str,
        supported_language: str,
        min_length: int,
        entropy_threshold: float,
        score: float,
        pattern: str,
    ) -> None:
        self._min_length = max(8, int(min_length))
        self._entropy_threshold = float(entropy_threshold)
        self._score = float(score)
        self._pattern_text = pattern
        super().__init__(
            supported_entities=["API_KEY"],
            name=name,
            supported_language=supported_language,
        )

    def load(self) -> None:
        self._candidate_re = re.compile(self._pattern_text)

    @staticmethod
    def _entropy(token: str) -> float:
        if not token:
            return 0.0
        total = len(token)
        counts: dict[str, int] = {}
        for char in token:
            counts[char] = counts.get(char, 0) + 1
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def analyze(self, text: str, entities: list[str], nlp_artifacts: Any = None) -> list[RecognizerResult]:
        if entities and "API_KEY" not in entities:
            return []

        results: list[RecognizerResult] = []
        seen: set[tuple[int, int]] = set()
        for match in self._candidate_re.finditer(text):
            token = match.group(0)
            if len(token) < self._min_length:
                continue
            if self._entropy(token) < self._entropy_threshold:
                continue
            start = int(match.start())
            end = int(match.end())
            if end <= start or (start, end) in seen:
                continue
            seen.add((start, end))
            results.append(
                RecognizerResult(
                    entity_type="API_KEY",
                    start=start,
                    end=end,
                    score=self._score,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                    },
                )
            )
        return results


def _pattern_flags(pattern_def: dict[str, Any]) -> int:
    flags_value = 0
    for flag in pattern_def.get("flags", []):
        flags_value |= _FLAG_MAP.get(str(flag).upper(), 0)
    return flags_value


def _build_regex_recognizers(
    recognizer_id: str,
    definition: RecognizerDefinition,
) -> list[EntityRecognizer]:
    params = definition.params
    patterns = params.get("patterns", [])
    recognizers: list[EntityRecognizer] = []
    by_label: dict[str, list[Pattern]] = {}
    flags_by_label: dict[str, int] = {}
    context_by_label: dict[str, list[str]] = {}
    for item in patterns:
        label = _normalize_entity_type(str(item.get("label", "")))
        if not label:
            continue
        pattern_name = str(item.get("name", recognizer_id))
        pattern_text = str(item.get("pattern", ""))
        if not pattern_text:
            continue
        pattern_score = float(item.get("score", params.get("score", 0.8)))
        by_label.setdefault(label, []).append(Pattern(name=pattern_name, regex=pattern_text, score=pattern_score))
        flags_by_label.setdefault(label, _pattern_flags(item))
        raw_context = item.get("context", params.get("context", []))
        if isinstance(raw_context, list):
            context_by_label.setdefault(label, [str(token) for token in raw_context])

    for label, label_patterns in by_label.items():
        recognizers.append(
            PatternRecognizer(
                supported_entity=label,
                name=f"{recognizer_id}:{label}",
                supported_language=_PRESIDIO_COMPAT_LANGUAGE,
                patterns=label_patterns,
                context=context_by_label.get(label),
                global_regex_flags=flags_by_label.get(label, re.IGNORECASE | re.MULTILINE | re.DOTALL),
            )
        )
    return recognizers


def _build_phone_recognizers(
    recognizer_id: str,
    definition: RecognizerDefinition,
) -> list[EntityRecognizer]:
    params = definition.params
    regions = [str(item) for item in params.get("regions", ["RU", "UA", "US"])]
    score = float(params.get("score", 0.92))
    min_digits = int(params.get("min_digits", 10))
    return [
        PhoneNumberRecognizer(
            name=recognizer_id,
            supported_language=_PRESIDIO_COMPAT_LANGUAGE,
            score=score,
            regions=regions,
            min_digits=min_digits,
        )
    ]


def _build_ip_recognizers(
    recognizer_id: str,
    definition: RecognizerDefinition,
) -> list[EntityRecognizer]:
    score = float(definition.params.get("score", 0.99))
    return [
        IPAddressRecognizer(
            name=recognizer_id,
            supported_language=_PRESIDIO_COMPAT_LANGUAGE,
            score=score,
        )
    ]


def _build_gliner_recognizers(
    recognizer_id: str,
    definition: RecognizerDefinition,
) -> list[EntityRecognizer]:
    params = definition.params
    labels = [str(item) for item in params.get("labels", [])]
    if not labels:
        return []
    model_name = str(params.get("model_name", "urchade/gliner_multi-v2.1"))
    threshold = float(params.get("threshold", 0.62))
    triton_model_name = str(params.get("triton_model_name", "gliner"))
    chunking = params.get("chunking", {})
    return [
        GlinerPresidioRecognizer(
            name=recognizer_id,
            supported_language=_PRESIDIO_COMPAT_LANGUAGE,
            model_name=model_name,
            labels=labels,
            threshold=threshold,
            triton_model_name=triton_model_name,
            chunking=chunking if isinstance(chunking, dict) else {},
        )
    ]


def _build_token_classifier_recognizers(
    recognizer_id: str,
    definition: RecognizerDefinition,
) -> list[EntityRecognizer]:
    if not settings.enable_nemotron:
        return []

    params = definition.params
    model_name = str(params.get("model_name", "scanpatch/pii-ner-nemotron"))
    threshold = float(params.get("threshold", 0.56))
    aggregation_strategy = str(params.get("aggregation_strategy", "simple"))
    triton_model_name = str(params.get("triton_model_name", "nemotron"))
    labels = [str(item) for item in params.get("labels", [])]
    raw_mapping = params.get("label_mapping", {})
    label_mapping = raw_mapping if isinstance(raw_mapping, dict) else {}
    raw_label_thresholds = params.get("raw_label_thresholds", {})
    entity_thresholds = params.get("entity_thresholds", {})
    chunking = params.get("chunking", {})
    return [
        TokenClassifierPresidioRecognizer(
            name=recognizer_id,
            supported_language=_PRESIDIO_COMPAT_LANGUAGE,
            model_name=model_name,
            threshold=threshold,
            labels=labels,
            label_mapping=label_mapping,
            raw_label_thresholds=raw_label_thresholds if isinstance(raw_label_thresholds, dict) else {},
            entity_thresholds=entity_thresholds if isinstance(entity_thresholds, dict) else {},
            aggregation_strategy=aggregation_strategy,
            triton_model_name=triton_model_name,
            chunking=chunking if isinstance(chunking, dict) else {},
        )
    ]


def _build_natasha_recognizers(
    recognizer_id: str,
    definition: RecognizerDefinition,
) -> list[EntityRecognizer]:
    params = definition.params
    score = float(params.get("score", 0.88))
    drop_prompt_imperatives = bool(params.get("drop_prompt_imperatives", True))
    min_person_chars = int(params.get("min_person_chars", 3))
    return [
        NatashaNerRecognizer(
            name=recognizer_id,
            supported_language=_PRESIDIO_COMPAT_LANGUAGE,
            score=score,
            drop_prompt_imperatives=drop_prompt_imperatives,
            min_person_chars=min_person_chars,
        )
    ]


def _build_recognizers_for_definition(
    recognizer_id: str,
    definition: RecognizerDefinition,
) -> list[EntityRecognizer]:
    rec_type = definition.type.lower()
    if rec_type in {"regex", "secret_regex"}:
        return _build_regex_recognizers(recognizer_id, definition)
    if rec_type == "phone":
        return _build_phone_recognizers(recognizer_id, definition)
    if rec_type == "ip":
        return _build_ip_recognizers(recognizer_id, definition)
    if rec_type == "gliner":
        return _build_gliner_recognizers(recognizer_id, definition)
    if rec_type == "token_classifier":
        return _build_token_classifier_recognizers(recognizer_id, definition)
    if rec_type == "natasha_ner":
        return _build_natasha_recognizers(recognizer_id, definition)
    if rec_type == "entropy":
        params = definition.params
        return [
            EntropySecretRecognizer(
                name=recognizer_id,
                supported_language=_PRESIDIO_COMPAT_LANGUAGE,
                min_length=int(params.get("min_length", 24)),
                entropy_threshold=float(params.get("entropy_threshold", 3.7)),
                score=float(params.get("score", 0.91)),
                pattern=str(params.get("pattern", r"\b[A-Za-z0-9_\-/+=]{20,}\b")),
            )
        ]
    return []


def build_recognizer_registry(
    *,
    recognizer_ids: list[str],
    recognizer_definitions: dict[str, RecognizerDefinition],
) -> RecognizerRegistry:
    registry = RecognizerRegistry()
    for recognizer_id in recognizer_ids:
        definition = recognizer_definitions.get(recognizer_id)
        if definition is None or not definition.enabled:
            continue
        for recognizer in _build_recognizers_for_definition(recognizer_id, definition):
            registry.add_recognizer(recognizer)
    return registry
