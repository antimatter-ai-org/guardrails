from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class TritonModelBinding:
    name: str
    infer_func: Callable[..., dict[str, Any]]
    inputs: list[Any]
    outputs: list[Any]
    config: Any
