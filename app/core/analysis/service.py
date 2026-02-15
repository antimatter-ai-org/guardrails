from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.config import AnalyzerProfile, PolicyConfig
from app.core.analysis.language import resolve_language, resolve_languages
from app.core.analysis.mapping import canonicalize_entity_type
from app.core.analysis.recognizers import build_recognizer_registry
from app.core.analysis.span_normalizer import normalize_detections
from app.models.entities import AnalysisDiagnostics, Detection
from app.settings import settings

logger = logging.getLogger(__name__)


class PresidioAnalysisService:
    _MAX_SAMPLE_ANALYSIS_SECONDS = 5.0
    _MAX_SAMPLE_DETECTIONS = 256
    _MAX_RAW_RECOGNIZER_RESULTS = 1024

    def __init__(self, policy_config: PolicyConfig) -> None:
        self._config = policy_config
        self._engines: dict[str, AnalyzerEngine] = {}
        self._registries: dict[str, RecognizerRegistry] = {}

    def _build_nlp_engine(self, profile: AnalyzerProfile) -> Any | None:
        analysis = profile.analysis
        if analysis.nlp_engine == "none":
            return None

        if analysis.nlp_engine != "transformers":
            raise ValueError(f"unsupported nlp_engine: {analysis.nlp_engine}")

        models = []
        for lang, model in analysis.nlp_models.items():
            if isinstance(model, dict):
                model_name = model
            else:
                model_name = {
                    "spacy": "en_core_web_sm" if lang == "en" else "ru_core_news_sm",
                    "transformers": str(model),
                }
            if settings.offline_mode:
                transformers_ref = model_name.get("transformers")
                if transformers_ref and transformers_ref.startswith("/"):
                    if not Path(transformers_ref).exists():
                        raise FileNotFoundError(f"transformers model path not found: {transformers_ref}")
                spacy_ref = model_name.get("spacy")
                if spacy_ref and spacy_ref.startswith("/"):
                    if not Path(spacy_ref).exists():
                        raise FileNotFoundError(f"spacy model path not found: {spacy_ref}")
            models.append(
                {
                    "lang_code": lang,
                    "model_name": model_name,
                }
            )

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "transformers",
                "models": models,
            }
        )
        return provider.create_engine()

    def _build_registry(self, profile_name: str) -> RecognizerRegistry:
        profile = self._config.analyzer_profiles.get(profile_name)
        if profile is None:
            raise KeyError(f"analyzer profile not found: {profile_name}")

        nlp_engine = self._build_nlp_engine(profile)
        return build_recognizer_registry(
            use_builtin_recognizers=profile.analysis.use_builtin_recognizers,
            supported_languages=profile.language.supported,
            recognizer_ids=profile.analysis.recognizers,
            recognizer_definitions=self._config.recognizer_definitions,
            nlp_engine=nlp_engine,
        )

    def _get_registry(self, profile_name: str) -> RecognizerRegistry:
        registry = self._registries.get(profile_name)
        if registry is not None:
            return registry
        registry = self._build_registry(profile_name)
        self._registries[profile_name] = registry
        return registry

    def _build_engine(self, profile_name: str) -> AnalyzerEngine:
        profile = self._config.analyzer_profiles.get(profile_name)
        if profile is None:
            raise KeyError(f"analyzer profile not found: {profile_name}")

        nlp_engine = self._build_nlp_engine(profile)
        registry = build_recognizer_registry(
            use_builtin_recognizers=profile.analysis.use_builtin_recognizers,
            supported_languages=profile.language.supported,
            recognizer_ids=profile.analysis.recognizers,
            recognizer_definitions=self._config.recognizer_definitions,
            nlp_engine=nlp_engine,
        )
        return AnalyzerEngine(
            registry=registry,
            nlp_engine=nlp_engine,
            supported_languages=profile.language.supported,
        )

    def _get_engine(self, profile_name: str) -> AnalyzerEngine:
        engine = self._engines.get(profile_name)
        if engine is not None:
            return engine
        engine = self._build_engine(profile_name)
        self._engines[profile_name] = engine
        return engine

    @staticmethod
    def _result_threshold(
        result: RecognizerResult,
        *,
        profile: AnalyzerProfile,
        policy_min_score: float,
    ) -> float:
        entity_type = result.entity_type.strip().upper()
        default_threshold = float(profile.analysis.thresholds.get("DEFAULT", policy_min_score))
        entity_threshold = float(profile.analysis.thresholds.get(entity_type, default_threshold))
        return max(float(policy_min_score), entity_threshold)

    @staticmethod
    def _requires_analyzer_engine(profile: AnalyzerProfile) -> bool:
        return bool(profile.analysis.use_builtin_recognizers or profile.analysis.nlp_engine != "none")

    def _analyze_with_registry(
        self,
        profile_name: str,
        text: str,
        language: str,
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
            if recognizer.get_supported_language() != language:
                continue
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
    def _merge_float_map(target: dict[str, float], source: dict[str, float]) -> None:
        for key, value in source.items():
            target[key] = target.get(key, 0.0) + float(value)

    @staticmethod
    def _merge_int_map(target: dict[str, int], source: dict[str, int]) -> None:
        for key, value in source.items():
            target[key] = target.get(key, 0) + int(value)

    def analyze_text(
        self,
        *,
        text: str,
        profile_name: str,
        policy_min_score: float,
        language_hint: str | None,
    ) -> tuple[str, list[Detection]]:
        language, detections, _ = self.analyze_text_with_diagnostics(
            text=text,
            profile_name=profile_name,
            policy_min_score=policy_min_score,
            language_hint=language_hint,
        )
        return language, detections

    def analyze_text_with_diagnostics(
        self,
        *,
        text: str,
        profile_name: str,
        policy_min_score: float,
        language_hint: str | None,
    ) -> tuple[str, list[Detection], AnalysisDiagnostics]:
        started_at = perf_counter()
        deadline = started_at + self._MAX_SAMPLE_ANALYSIS_SECONDS
        profile = self._config.analyzer_profiles.get(profile_name)
        if profile is None:
            raise KeyError(f"analyzer profile not found: {profile_name}")

        languages = self.resolve_languages(
            text=text,
            profile_name=profile_name,
            language_hint=language_hint,
        )
        language = languages[0]
        detector_timing_ms: dict[str, float] = {}
        detector_span_counts: dict[str, int] = {}
        detector_errors: dict[str, str] = {}
        timeout_exceeded = False
        spans_truncated = False

        results_with_language: list[tuple[str, RecognizerResult]] = []
        if self._requires_analyzer_engine(profile):
            engine = self._get_engine(profile_name)
            for current_language in languages:
                if perf_counter() >= deadline:
                    timeout_exceeded = True
                    break
                recognizer_name = f"analyzer_engine:{current_language}"
                call_started = perf_counter()
                try:
                    results = engine.analyze(
                        text=text,
                        language=current_language,
                        score_threshold=float(policy_min_score),
                        return_decision_process=False,
                    )
                except Exception as exc:
                    results = []
                    detector_errors[recognizer_name] = str(exc)
                    logger.warning("analyzer engine failed (%s): %s", recognizer_name, exc)
                detector_timing_ms[recognizer_name] = (perf_counter() - call_started) * 1000.0
                detector_span_counts[recognizer_name] = len(results)
                if perf_counter() >= deadline:
                    timeout_exceeded = True
                for result in results:
                    results_with_language.append((current_language, result))
        else:
            for current_language in languages:
                if perf_counter() >= deadline:
                    timeout_exceeded = True
                    break
                (
                    results,
                    language_timing,
                    language_spans,
                    language_errors,
                    language_timeout_exceeded,
                ) = self._analyze_with_registry(profile_name, text, current_language, deadline=deadline)
                self._merge_float_map(detector_timing_ms, language_timing)
                self._merge_int_map(detector_span_counts, language_spans)
                detector_errors.update(language_errors)
                timeout_exceeded = timeout_exceeded or language_timeout_exceeded
                for result in results:
                    results_with_language.append((current_language, result))

        deduped_results: list[tuple[str, RecognizerResult]] = []
        dedup_index: dict[tuple[int, int, str], int] = {}
        for current_language, result in results_with_language:
            key = (int(result.start), int(result.end), result.entity_type.strip().upper())
            existing_index = dedup_index.get(key)
            if existing_index is None:
                dedup_index[key] = len(deduped_results)
                deduped_results.append((current_language, result))
                continue
            _, previous_result = deduped_results[existing_index]
            if float(result.score) > float(previous_result.score):
                deduped_results[existing_index] = (current_language, result)

        detections: list[Detection] = []
        for result_language, result in deduped_results:
            threshold = self._result_threshold(result, profile=profile, policy_min_score=policy_min_score)
            if float(result.score) < threshold:
                continue

            entity_type = result.entity_type.strip().upper()
            canonical = canonicalize_entity_type(entity_type, custom_mapping=profile.analysis.label_mapping)
            metadata = dict(result.recognition_metadata or {})
            metadata["entity_type"] = entity_type
            metadata["canonical_label"] = canonical
            metadata["language"] = result_language
            metadata["analysis_languages"] = languages

            label = canonical.upper() if canonical else entity_type
            detections.append(
                Detection(
                    start=int(result.start),
                    end=int(result.end),
                    text=text[int(result.start) : int(result.end)],
                    label=label,
                    score=float(result.score),
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
        return language, detections, diagnostics

    def resolve_language(
        self,
        *,
        text: str,
        profile_name: str,
        language_hint: str | None,
    ) -> str:
        profile = self._config.analyzer_profiles.get(profile_name)
        if profile is None:
            raise KeyError(f"analyzer profile not found: {profile_name}")
        return resolve_language(
            text=text,
            language_hint=language_hint,
            supported=profile.language.supported,
            default_language=profile.language.default,
            detection_mode=profile.language.detection,
        )

    def resolve_languages(
        self,
        *,
        text: str,
        profile_name: str,
        language_hint: str | None,
    ) -> list[str]:
        profile = self._config.analyzer_profiles.get(profile_name)
        if profile is None:
            raise KeyError(f"analyzer profile not found: {profile_name}")
        return resolve_languages(
            text=text,
            language_hint=language_hint,
            supported=profile.language.supported,
            default_language=profile.language.default,
            detection_mode=profile.language.detection,
            strategy="union",
            union_min_share=0.2,
        )
