from __future__ import annotations

from types import SimpleNamespace

from app.core.analysis.recognizers import NatashaNerRecognizer


class _FakeDoc:
    def __init__(self, text: str) -> None:
        self.text = text
        self.spans = [
            SimpleNamespace(type="PER", start=0, stop=9),
            SimpleNamespace(type="PER", start=10, stop=29),
            SimpleNamespace(type="ORG", start=33, stop=42),
            SimpleNamespace(type="LOC", start=46, stop=52),
        ]

    def segment(self, segmenter: object) -> None:
        return None

    def tag_ner(self, tagger: object) -> None:
        return None


def test_natasha_recognizer_maps_entities_and_drops_prompt_verb() -> None:
    text = "Перескажи Ваня Миллипиздриков, Acme Corp, Москва."
    recognizer = NatashaNerRecognizer(
        name="natasha_ner_ru",
        supported_language="global",
        score=0.88,
        drop_prompt_imperatives=True,
        min_person_chars=3,
    )
    recognizer._segmenter = object()  # noqa: SLF001
    recognizer._tagger = object()  # noqa: SLF001
    recognizer._doc_cls = _FakeDoc  # noqa: SLF001

    results = recognizer.analyze(text, entities=["PERSON", "ORGANIZATION", "LOCATION"])

    entity_types = [item.entity_type for item in results]
    spans = [(item.start, item.end) for item in results]
    assert entity_types == ["PERSON", "ORGANIZATION", "LOCATION"]
    assert (0, 9) not in spans
    assert (10, 29) in spans


def test_natasha_recognizer_respects_entity_filter() -> None:
    text = "Перескажи Ваня Миллипиздриков, Acme Corp, Москва."
    recognizer = NatashaNerRecognizer(
        name="natasha_ner_ru",
        supported_language="global",
        score=0.88,
    )
    recognizer._segmenter = object()  # noqa: SLF001
    recognizer._tagger = object()  # noqa: SLF001
    recognizer._doc_cls = _FakeDoc  # noqa: SLF001

    results = recognizer.analyze(text, entities=["ORGANIZATION"])

    assert len(results) == 1
    assert results[0].entity_type == "ORGANIZATION"
