from __future__ import annotations

from app.detectors.base import Detector
from app.models.entities import Detection


class NatashaDetector(Detector):
    def __init__(
        self,
        name: str,
        score: float = 0.7,
        embedding_path: str | None = None,
        ner_path: str | None = None,
    ) -> None:
        super().__init__(name)
        try:
            from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
        except Exception as exc:  # pragma: no cover - import path varies by env
            raise RuntimeError("natasha is not installed. Install project dependencies (for example: `uv sync`).") from exc

        self._doc_cls = Doc
        self._segmenter = Segmenter()
        embedding = NewsEmbedding(path=embedding_path) if embedding_path else NewsEmbedding()
        self._tagger = NewsNERTagger(embedding, path=ner_path) if ner_path else NewsNERTagger(embedding)
        self._score = score

    def detect(self, text: str) -> list[Detection]:
        doc = self._doc_cls(text)
        doc.segment(self._segmenter)
        doc.tag_ner(self._tagger)

        findings: list[Detection] = []
        for span in doc.spans:
            findings.append(
                Detection(
                    start=span.start,
                    end=span.stop,
                    text=text[span.start : span.stop],
                    label=f"NER_{span.type}",
                    score=self._score,
                    detector=self.name,
                    metadata={"source": "natasha"},
                )
            )
        return findings
