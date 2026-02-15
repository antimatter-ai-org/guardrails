from __future__ import annotations

from app.pytriton_server import registry


class _StubModel:
    def __init__(self, **kwargs):
        self._name = kwargs["triton_model_name"]

    def binding(self):
        return self._name


def test_build_bindings_without_nemotron(monkeypatch):
    monkeypatch.setattr(registry, "GlinerTritonModel", _StubModel)
    monkeypatch.setattr(registry, "TokenClassifierTritonModel", _StubModel)
    bindings = registry.build_bindings(
        gliner_model_ref="urchade/gliner_multi-v2.1",
        token_classifier_model_ref="scanpatch/pii-ner-nemotron",
        device="cuda",
        max_batch_size=32,
        enable_nemotron=False,
    )
    assert bindings == ["gliner"]


def test_build_bindings_with_nemotron(monkeypatch):
    monkeypatch.setattr(registry, "GlinerTritonModel", _StubModel)
    monkeypatch.setattr(registry, "TokenClassifierTritonModel", _StubModel)
    bindings = registry.build_bindings(
        gliner_model_ref="urchade/gliner_multi-v2.1",
        token_classifier_model_ref="scanpatch/pii-ner-nemotron",
        device="cuda",
        max_batch_size=32,
        enable_nemotron=True,
    )
    assert bindings == ["gliner", "nemotron"]
