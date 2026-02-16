from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import regex as re
from presidio_analyzer import EntityRecognizer, RecognizerRegistry, RecognizerResult

from app.config import PolicyConfig
from app.core.analysis.mapping import canonicalize_entity_type
from app.core.analysis.recognizers import build_recognizer_registry
from app.core.analysis.span_normalizer import normalize_detections
from app.models.entities import AnalysisDiagnostics, Detection

logger = logging.getLogger(__name__)


class PresidioAnalysisService:
    _MAX_SAMPLE_ANALYSIS_SECONDS = 5.0
    _MAX_SAMPLE_DETECTIONS = 256
    _MAX_RAW_RECOGNIZER_RESULTS = 1024
    _PAYMENT_CARD_CONTEXT_MARKERS = (
        "card",
        "credit",
        "debit",
        "visa",
        "mastercard",
        "amex",
        "payment",
        "карта",
        "номер карты",
        "банковск",
        "платеж",
    )

    def __init__(self, policy_config: PolicyConfig) -> None:
        self._config = policy_config
        self._registries: dict[str, RecognizerRegistry] = {
            profile_name: self._build_registry(profile_name)
            for profile_name in sorted(self._config.analyzer_profiles.keys())
        }

    def _build_registry(self, profile_name: str) -> RecognizerRegistry:
        profile = self._config.analyzer_profiles.get(profile_name)
        if profile is None:
            raise KeyError(f"analyzer profile not found: {profile_name}")

        return build_recognizer_registry(
            recognizer_ids=profile.analysis.recognizers,
            recognizer_definitions=self._config.recognizer_definitions,
        )

    def _get_registry(self, profile_name: str) -> RecognizerRegistry:
        registry = self._registries.get(profile_name)
        if registry is not None:
            return registry
        raise KeyError(f"analyzer profile registry not initialized: {profile_name}")

    def ensure_profile_runtimes_ready(self, *, profile_names: list[str], timeout_s: float) -> dict[str, str]:
        errors: dict[str, str] = {}
        readiness_timeout = max(0.0, float(timeout_s))
        for profile_name in profile_names:
            registry = self._get_registry(profile_name)
            for recognizer in registry.recognizers:
                runtime = getattr(recognizer, "_runtime", None)
                if runtime is None:
                    continue
                recognizer_name = str(getattr(recognizer, "name", recognizer.__class__.__name__))
                runtime_key = f"{profile_name}:{recognizer_name}"
                try:
                    if hasattr(runtime, "ensure_ready"):
                        ready = bool(runtime.ensure_ready(timeout_s=readiness_timeout))
                    elif hasattr(runtime, "is_ready"):
                        ready = bool(runtime.is_ready())
                    else:
                        continue
                except Exception as exc:
                    errors[runtime_key] = str(exc)
                    continue
                if ready:
                    continue
                load_error = None
                if hasattr(runtime, "load_error"):
                    try:
                        load_error = runtime.load_error()
                    except Exception:
                        load_error = None
                errors[runtime_key] = str(load_error or "runtime is not ready")
        return errors

    @staticmethod
    def _result_threshold(*, policy_min_score: float) -> float:
        return float(policy_min_score)

    def _analyze_with_registry(
        self,
        profile_name: str,
        text: str,
        *,
        deadline: float,
    ) -> tuple[list[RecognizerResult], dict[str, float], dict[str, int], dict[str, str], bool]:
        registry = self._get_registry(profile_name)
        raw_results: list[RecognizerResult] = []
        detector_timing_ms: dict[str, float] = {}
        detector_span_counts: dict[str, int] = {}
        detector_errors: dict[str, str] = {}
        timeout_exceeded = False

        for recognizer in registry.recognizers:
            if perf_counter() >= deadline:
                timeout_exceeded = True
                break
            entities = recognizer.get_supported_entities()
            recognizer_name = str(getattr(recognizer, "name", recognizer.__class__.__name__))
            started_at = perf_counter()
            try:
                recognized = recognizer.analyze(text=text, entities=entities, nlp_artifacts=None)
            except Exception as exc:
                recognized = []
                detector_errors[recognizer_name] = str(exc)
                logger.warning("recognizer failed (%s): %s", recognizer_name, exc)
            elapsed_ms = (perf_counter() - started_at) * 1000.0
            detector_timing_ms[recognizer_name] = detector_timing_ms.get(recognizer_name, 0.0) + elapsed_ms
            detector_span_counts[recognizer_name] = detector_span_counts.get(recognizer_name, 0) + len(recognized)
            raw_results.extend(recognized)
            if perf_counter() >= deadline:
                timeout_exceeded = True
                break
            if len(raw_results) >= self._MAX_RAW_RECOGNIZER_RESULTS:
                break

        return (
            EntityRecognizer.remove_duplicates(raw_results),
            detector_timing_ms,
            detector_span_counts,
            detector_errors,
            timeout_exceeded,
        )

    @staticmethod
    def _luhn_check(digits: str) -> bool:
        if not digits or not digits.isdigit():
            return False
        total = 0
        parity = len(digits) % 2
        for idx, char in enumerate(digits):
            value = int(char)
            if idx % 2 == parity:
                value *= 2
                if value > 9:
                    value -= 9
            total += value
        return total % 10 == 0

    @classmethod
    def _payment_card_signal(cls, *, text: str, start: int, end: int) -> dict[str, Any]:
        if end <= start:
            return {
                "keep": False,
                "digits_count": 0,
                "luhn_valid": False,
                "grouped_like_card": False,
                "has_context": False,
                "score_multiplier": 1.0,
                "score_bonus": 0.0,
            }
        snippet = text[start:end]
        digits = "".join(char for char in snippet if char.isdigit())
        grouped_like_card = bool(re.fullmatch(r"(?:\d{4}[ -]){2,4}\d{1,4}", re.sub(r"\s+", " ", snippet.strip())))
        context_start = max(0, start - 32)
        context_end = min(len(text), end + 32)
        context = text[context_start:context_end].lower()
        has_context = any(marker in context for marker in cls._PAYMENT_CARD_CONTEXT_MARKERS)
        luhn_valid = cls._luhn_check(digits)
        if len(digits) < 13 or len(digits) > 19:
            return {
                "keep": False,
                "digits_count": len(digits),
                "luhn_valid": luhn_valid,
                "grouped_like_card": grouped_like_card,
                "has_context": has_context,
                "score_multiplier": 1.0,
                "score_bonus": 0.0,
            }
        if len(set(digits)) < 2:
            return {
                "keep": False,
                "digits_count": len(digits),
                "luhn_valid": luhn_valid,
                "grouped_like_card": grouped_like_card,
                "has_context": has_context,
                "score_multiplier": 1.0,
                "score_bonus": 0.0,
            }
        weak_continuous = not grouped_like_card and not luhn_valid and not has_context
        score_multiplier = 0.9 if weak_continuous else 1.0
        score_bonus = 0.0
        if luhn_valid:
            score_bonus = 0.08
        elif grouped_like_card and has_context:
            score_bonus = 0.03
        return {
            "keep": True,
            "digits_count": len(digits),
            "luhn_valid": luhn_valid,
            "grouped_like_card": grouped_like_card,
            "has_context": has_context,
            "score_multiplier": score_multiplier,
            "score_bonus": score_bonus,
        }

    def analyze_text(
        self,
        *,
        text: str,
        profile_name: str,
        policy_min_score: float,
    ) -> list[Detection]:
        detections, _ = self.analyze_text_with_diagnostics(
            text=text,
            profile_name=profile_name,
            policy_min_score=policy_min_score,
        )
        return detections

    def analyze_text_with_diagnostics(
        self,
        *,
        text: str,
        profile_name: str,
        policy_min_score: float,
    ) -> tuple[list[Detection], AnalysisDiagnostics]:
        started_at = perf_counter()
        deadline = started_at + self._MAX_SAMPLE_ANALYSIS_SECONDS
        detector_timing_ms: dict[str, float] = {}
        detector_span_counts: dict[str, int] = {}
        detector_errors: dict[str, str] = {}
        timeout_exceeded = False
        spans_truncated = False

        (
            results,
            detector_timing_ms,
            detector_span_counts,
            detector_errors,
            timeout_exceeded,
        ) = self._analyze_with_registry(profile_name, text, deadline=deadline)

        detections: list[Detection] = []
        for result in results:
            entity_type = result.entity_type.strip().upper()
            canonical = canonicalize_entity_type(entity_type)
            score = float(result.score)
            card_signal: dict[str, Any] | None = None
            if canonical == "payment_card":
                card_signal = self._payment_card_signal(
                    text=text,
                    start=int(result.start),
                    end=int(result.end),
                )
                if not bool(card_signal.get("keep")):
                    continue
                detector_name = str((result.recognition_metadata or {}).get("recognizer_name", ""))
                if ":CREDIT_CARD" in detector_name:
                    score *= float(card_signal.get("score_multiplier", 1.0))
                score = min(1.0, score + float(card_signal.get("score_bonus", 0.0)))

            threshold = self._result_threshold(policy_min_score=policy_min_score)
            if score < threshold:
                continue
            metadata = dict(result.recognition_metadata or {})
            metadata["entity_type"] = entity_type
            metadata["canonical_label"] = canonical
            if card_signal is not None:
                metadata["payment_card_signal"] = card_signal

            label = canonical.upper() if canonical else entity_type
            detections.append(
                Detection(
                    start=int(result.start),
                    end=int(result.end),
                    text=text[int(result.start) : int(result.end)],
                    label=label,
                    score=score,
                    detector=str(metadata.get("recognizer_name", "presidio")),
                    metadata=metadata,
                )
            )

        detections, postprocess_stats = normalize_detections(
            text=text,
            detections=detections,
            return_stats=True,
        )
        if len(detections) > self._MAX_SAMPLE_DETECTIONS:
            spans_truncated = True
            detections = sorted(
                detections,
                key=lambda item: (-float(item.score), item.start, -(item.end - item.start)),
            )[: self._MAX_SAMPLE_DETECTIONS]
            detections = sorted(detections, key=lambda item: item.start)

        diagnostics = AnalysisDiagnostics(
            elapsed_ms=(perf_counter() - started_at) * 1000.0,
            detector_timing_ms={key: round(value, 3) for key, value in detector_timing_ms.items()},
            detector_span_counts=dict(detector_span_counts),
            detector_errors=dict(detector_errors),
            postprocess_mutations=dict(postprocess_stats),
            limit_flags={
                "analysis_timeout_exceeded": bool(timeout_exceeded),
                "max_spans_truncated": bool(spans_truncated),
            },
        )
        return detections, diagnostics
