from __future__ import annotations

from abc import ABC, abstractmethod

from app.eval.types import EvalSample


class DatasetAdapter(ABC):
    @property
    def supports_synthetic_split(self) -> bool:
        return False

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_samples(
        self,
        split: str,
        cache_dir: str,
        hf_token: str | None,
        synthetic_test_size: float = 0.2,
        synthetic_split_seed: int = 42,
        max_samples: int | None = None,
    ) -> list[EvalSample]:
        raise NotImplementedError
