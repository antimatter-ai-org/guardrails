from __future__ import annotations

from pathlib import Path
from typing import Any

from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.config import AnalyzerProfile, PolicyConfig
from app.core.analysis.language import resolve_language
from app.core.analysis.mapping import canonicalize_entity_type
from app.core.analysis.recognizers import build_recognizer_registry
from app.models.entities import Detection
from app.settings import settings


class PresidioAnalysisService:
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

    def _analyze_with_registry(self, profile_name: str, text: str, language: str) -> list[RecognizerResult]:
        registry = self._get_registry(profile_name)
        raw_results: list[RecognizerResult] = []

        for recognizer in registry.recognizers:
            if recognizer.get_supported_language() != language:
                continue
            entities = recognizer.get_supported_entities()
            raw_results.extend(recognizer.analyze(text=text, entities=entities, nlp_artifacts=None))

        return EntityRecognizer.remove_duplicates(raw_results)

    def analyze_text(
        self,
        *,
        text: str,
        profile_name: str,
        policy_min_score: float,
        language_hint: str | None,
    ) -> tuple[str, list[Detection]]:
        profile = self._config.analyzer_profiles.get(profile_name)
        if profile is None:
            raise KeyError(f"analyzer profile not found: {profile_name}")

        language = self.resolve_language(
            text=text,
            profile_name=profile_name,
            language_hint=language_hint,
        )

        if self._requires_analyzer_engine(profile):
            engine = self._get_engine(profile_name)
            results = engine.analyze(
                text=text,
                language=language,
                score_threshold=float(policy_min_score),
                return_decision_process=False,
            )
        else:
            results = self._analyze_with_registry(profile_name, text, language)

        detections: list[Detection] = []
        for result in results:
            threshold = self._result_threshold(result, profile=profile, policy_min_score=policy_min_score)
            if float(result.score) < threshold:
                continue

            entity_type = result.entity_type.strip().upper()
            canonical = canonicalize_entity_type(entity_type, custom_mapping=profile.analysis.label_mapping)
            metadata = dict(result.recognition_metadata or {})
            metadata["entity_type"] = entity_type
            metadata["canonical_label"] = canonical
            metadata["language"] = language

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

        return language, detections

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
