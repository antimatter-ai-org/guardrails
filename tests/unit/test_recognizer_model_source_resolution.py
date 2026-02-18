from __future__ import annotations

from app.config import RecognizerDefinition
from app.core.analysis import recognizers


def test_gliner_builder_uses_resolved_model_source(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_resolve(*, model_name: str, model_dir: str | None, strict: bool) -> str:
        captured["resolve_model_name"] = model_name
        captured["resolve_model_dir"] = model_dir
        captured["resolve_strict"] = strict
        return "/models/gliner/local"

    def fake_recognizer(**kwargs):  # noqa: ANN003
        captured["recognizer_model_name"] = kwargs["model_name"]
        return object()

    monkeypatch.setattr(recognizers.settings, "model_dir", "/models")
    monkeypatch.setattr(recognizers.settings, "offline_mode", True)
    monkeypatch.setattr(recognizers.settings, "enable_gliner", True)
    monkeypatch.setattr(recognizers, "resolve_gliner_model_source", fake_resolve)
    monkeypatch.setattr(recognizers, "GlinerPresidioRecognizer", fake_recognizer)

    definition = RecognizerDefinition(
        type="gliner",
        enabled=True,
        params={"model_name": "urchade/gliner_multi-v2.1", "labels": ["person"]},
    )
    built = recognizers._build_gliner_recognizers("gliner_pii_multilingual", definition)  # noqa: SLF001
    assert len(built) == 1
    assert captured["resolve_model_name"] == "urchade/gliner_multi-v2.1"
    assert captured["resolve_model_dir"] == "/models"
    assert captured["resolve_strict"] is True
    assert captured["recognizer_model_name"] == "/models/gliner/local"


def test_gliner_builder_respects_global_enable_knob(monkeypatch) -> None:
    definition = RecognizerDefinition(
        type="gliner",
        enabled=True,
        params={"model_name": "urchade/gliner_multi-v2.1", "labels": ["person"]},
    )
    monkeypatch.setattr(recognizers.settings, "enable_gliner", False)
    assert recognizers._build_gliner_recognizers("gliner", definition) == []  # noqa: SLF001


def test_token_classifier_builder_uses_resolved_model_source(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_resolve(*, model_name: str, model_dir: str | None, strict: bool) -> str:
        captured["resolve_model_name"] = model_name
        captured["resolve_model_dir"] = model_dir
        captured["resolve_strict"] = strict
        return "/models/token_classifier/local"

    def fake_recognizer(**kwargs):  # noqa: ANN003
        captured["recognizer_model_name"] = kwargs["model_name"]
        return object()

    monkeypatch.setattr(recognizers.settings, "enable_nemotron", True)
    monkeypatch.setattr(recognizers.settings, "model_dir", "/models")
    monkeypatch.setattr(recognizers.settings, "offline_mode", False)
    monkeypatch.setattr(recognizers, "resolve_token_classifier_model_source", fake_resolve)
    monkeypatch.setattr(recognizers, "TokenClassifierPresidioRecognizer", fake_recognizer)

    definition = RecognizerDefinition(
        type="token_classifier",
        enabled=True,
        params={"model_name": "scanpatch/pii-ner-nemotron", "labels": ["email"]},
    )
    built = recognizers._build_token_classifier_recognizers("nemotron", definition)  # noqa: SLF001
    assert len(built) == 1
    assert captured["resolve_model_name"] == "scanpatch/pii-ner-nemotron"
    assert captured["resolve_model_dir"] == "/models"
    assert captured["resolve_strict"] is False
    assert captured["recognizer_model_name"] == "/models/token_classifier/local"
