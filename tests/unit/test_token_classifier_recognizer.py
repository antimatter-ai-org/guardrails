from __future__ import annotations

from app.config import RecognizerDefinition
from app.core.analysis import recognizers


class _StubRuntime:
    def __init__(self, payload):
        self._payload = payload

    def predict_entities(self, text: str, labels: list[str], threshold: float):
        return list(self._payload)


def test_token_classifier_recognizer_applies_entity_threshold(monkeypatch):
    payload = [
        {"start": 0, "end": 4, "label": "email", "score": 0.61},
        {"start": 5, "end": 10, "label": "email", "score": 0.8},
    ]
    monkeypatch.setattr(
        recognizers,
        "build_token_classifier_runtime",
        lambda **kwargs: _StubRuntime(payload),
    )
    rec = recognizers.TokenClassifierPresidioRecognizer(
        name="test",
        supported_language="en",
        model_name="scanpatch/pii-ner-nemotron",
        threshold=0.5,
        labels=["email"],
        label_mapping={"email": "EMAIL_ADDRESS"},
        entity_thresholds={"EMAIL_ADDRESS": 0.7},
    )
    out = rec.analyze("abcd efghij", entities=["EMAIL_ADDRESS"])
    assert len(out) == 1
    assert out[0].start == 5
    assert out[0].end == 10


def test_token_classifier_recognizer_applies_raw_label_threshold(monkeypatch):
    payload = [
        {"start": 0, "end": 4, "label": "B-ip", "score": 0.7},
        {"start": 5, "end": 13, "label": "ip", "score": 0.9},
    ]
    monkeypatch.setattr(
        recognizers,
        "build_token_classifier_runtime",
        lambda **kwargs: _StubRuntime(payload),
    )
    rec = recognizers.TokenClassifierPresidioRecognizer(
        name="test",
        supported_language="en",
        model_name="scanpatch/pii-ner-nemotron",
        threshold=0.5,
        labels=["ip"],
        raw_label_thresholds={"ip": 0.8},
    )
    out = rec.analyze("1.1.1.1 8.8.8.8", entities=["IP_ADDRESS"])
    assert len(out) == 1
    assert out[0].start == 5
    assert out[0].end == 13


def test_token_classifier_builder_respects_global_enable_knob(monkeypatch):
    definition = RecognizerDefinition(
        type="token_classifier",
        enabled=True,
        params={"model_name": "scanpatch/pii-ner-nemotron", "labels": ["email"]},
    )
    monkeypatch.setattr(recognizers.settings, "enable_nemotron", False)
    assert recognizers._build_token_classifier_recognizers("nemotron", definition, ["en"]) == []  # noqa: SLF001
