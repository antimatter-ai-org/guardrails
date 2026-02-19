from __future__ import annotations

from app.pytriton_server import registry


class _StubModel:
    def __init__(self, **kwargs):
        self._name = kwargs["triton_model_name"]

    def binding(self):
        return self._name


def test_build_bindings_with_nemotron(monkeypatch) -> None:
    monkeypatch.setattr(registry, "TokenClassifierTritonModel", _StubModel)
    bindings = registry.build_bindings(
        token_classifier_model_ref="scanpatch/pii-ner-nemotron",
        device="cuda",
        max_batch_size=32,
        enable_nemotron=True,
    )
    assert bindings == ["nemotron"]


def test_build_bindings_without_models() -> None:
    bindings = registry.build_bindings(
        token_classifier_model_ref="scanpatch/pii-ner-nemotron",
        device="cuda",
        max_batch_size=32,
        enable_nemotron=False,
    )
    assert bindings == []
