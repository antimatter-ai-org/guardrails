from __future__ import annotations

from app.eval.datasets.base import DatasetAdapter
from app.eval.datasets.scanpatch import ScanpatchSyntheticControlledAdapter

_ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "scanpatch/pii-ner-corpus-synthetic-controlled": ScanpatchSyntheticControlledAdapter,
}


def get_dataset_adapter(dataset_name: str) -> DatasetAdapter:
    adapter_cls = _ADAPTERS.get(dataset_name)
    if adapter_cls is None:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")
    return adapter_cls()
