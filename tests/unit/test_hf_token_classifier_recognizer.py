from __future__ import annotations

import sys
import time
import types

from app.config import RecognizerDefinition
from app.core.analysis import recognizers as recognizers_module
from app.core.analysis.recognizers import HFTokenClassifierRecognizer, _build_recognizers_for_definition


def test_hf_token_classifier_analyze_maps_and_filters_entities(monkeypatch) -> None:
    monkeypatch.setattr(HFTokenClassifierRecognizer, "load", lambda self: None)

    recognizer = HFTokenClassifierRecognizer(
        name="hf_ner:en",
        supported_language="en",
        model_name="dslim/bert-base-NER",
        score_threshold=0.5,
        label_mapping={"PER": "PERSON", "ORG": "ORGANIZATION"},
        entities=["PERSON", "ORGANIZATION"],
        aggregation_strategy="simple",
    )
    recognizer._pipeline = lambda _: [  # type: ignore[assignment]
        {"entity_group": "PER", "score": 0.93, "start": 0, "end": 5},
        {"entity_group": "ORG", "score": 0.97, "start": 10, "end": 14},
    ]

    results = recognizer.analyze("Alice at ACME", entities=["PERSON"])

    assert len(results) == 1
    assert results[0].entity_type == "PERSON"
    assert results[0].start == 0
    assert results[0].end == 5


def test_hf_token_classifier_builder_resolves_model_source(monkeypatch) -> None:
    monkeypatch.setattr(HFTokenClassifierRecognizer, "load", lambda self: None)
    monkeypatch.setattr(
        recognizers_module,
        "resolve_hf_model_source",
        lambda **_: "/models/hf_token_classifier/dslim__bert-base-NER",
    )

    definition = RecognizerDefinition(
        type="hf_token_classifier",
        enabled=True,
        params={
            "model_name": "dslim/bert-base-NER",
            "entities": ["PERSON"],
            "label_mapping": {"PER": "PERSON"},
        },
    )

    recognizers = _build_recognizers_for_definition("hf_ner", definition, ["en"])

    assert len(recognizers) == 1
    assert isinstance(recognizers[0], HFTokenClassifierRecognizer)
    assert recognizers[0]._model_name == "/models/hf_token_classifier/dslim__bert-base-NER"  # noqa: SLF001


def test_hf_token_classifier_load_falls_back_to_cpu_when_mps_init_fails(monkeypatch) -> None:
    calls: list[object] = []

    def fake_pipeline(**kwargs):
        device = kwargs.get("device")
        calls.append(device)
        if device == "mps":
            raise RuntimeError("mps init failed")
        return lambda _: []

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=fake_pipeline))
    monkeypatch.setattr(recognizers_module, "resolve_cpu_runtime_device", lambda _: "mps")
    monkeypatch.setattr(recognizers_module.settings, "runtime_mode", "cpu")
    monkeypatch.setattr(recognizers_module.settings, "cpu_device", "auto")

    recognizer = HFTokenClassifierRecognizer(
        name="hf_ner:en",
        supported_language="en",
        model_name="dslim/bert-base-NER",
        score_threshold=0.5,
        label_mapping={},
        entities=["PERSON"],
        aggregation_strategy="simple",
    )

    assert recognizer._pipeline is not None  # noqa: SLF001
    assert calls == ["mps", -1]


def test_hf_token_classifier_timeout_gracefully_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(HFTokenClassifierRecognizer, "load", lambda self: None)

    recognizer = HFTokenClassifierRecognizer(
        name="hf_ner:en",
        supported_language="en",
        model_name="dslim/bert-base-NER",
        score_threshold=0.1,
        label_mapping={},
        entities=["PERSON"],
        aggregation_strategy="simple",
        infer_timeout_seconds=0.01,
    )

    def slow_pipeline(_: str) -> list[dict[str, object]]:
        time.sleep(0.1)
        return [{"entity_group": "PER", "score": 0.99, "start": 0, "end": 5}]

    recognizer._pipeline = slow_pipeline  # type: ignore[assignment]

    assert recognizer.analyze("Alice", entities=["PERSON"]) == []
