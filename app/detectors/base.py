from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.entities import Detection


class Detector(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def detect(self, text: str) -> list[Detection]:
        raise NotImplementedError
