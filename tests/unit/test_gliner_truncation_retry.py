from __future__ import annotations

import warnings

import pytest

from app.runtime.gliner_runtime import LocalCpuGlinerRuntime
from app.runtime.gliner_word_chunking import split_gliner_words


def test_local_gliner_runtime_retries_on_truncation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class _StubModel:
        def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict]:
            word_len = len(split_gliner_words(text))
            calls.append(word_len)
            if word_len > 4:
                warnings.warn(
                    "Sentence of length 999 has been truncated to 384",
                    UserWarning,
                    stacklevel=2,
                )
            return []

    class _StubTokenizer:
        is_fast = True

        def __call__(self, text: str, add_special_tokens: bool = True, truncation: bool = False):  # noqa: ANN001
            # Cheap, deterministic stand-in: token count ~= whitespace word count.
            return {"input_ids": [0] * max(1, len(str(text).split()))}

    def fake_load(self) -> None:  # noqa: ANN001
        self.device = "cpu"  # noqa: SLF001
        self._model = _StubModel()  # noqa: SLF001
        self._encoder_tokenizer = _StubTokenizer()  # noqa: SLF001
        self._encoder_max_len = 10_000  # noqa: SLF001
        self._gliner_cfg = {  # noqa: SLF001
            "max_len": 16,
            "max_width": 1,
            "ent_token": "<ENT>",
            "sep_token": "<SEP>",
            "model_name": "dummy-encoder",
        }
        self._load_error = None  # noqa: SLF001

    monkeypatch.setattr(LocalCpuGlinerRuntime, "_load_model", fake_load)
    runtime = LocalCpuGlinerRuntime(model_name="dummy")
    runtime.predict_entities(" ".join(["w"] * 40), labels=["person"], threshold=0.5)

    assert any(value > 4 for value in calls), "expected at least one truncation-triggering chunk"
    assert max(calls[-10:]) <= 4, "expected retries to reduce chunk sizes below truncation threshold"

