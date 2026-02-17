from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvalCachePaths:
    root: Path
    hf: Path
    views: Path
    collections: Path

    @classmethod
    def from_cache_dir_arg(cls, cache_dir: str) -> "EvalCachePaths":
        """
        Historically, the evaluator used --cache-dir=.eval_cache/hf.
        The redesigned evaluator treats --cache-dir as an eval root and nests
        HF caches under <root>/hf.

        This helper accepts both:
        - .eval_cache          -> root=.eval_cache, hf=.eval_cache/hf
        - .eval_cache/hf       -> root=.eval_cache, hf=.eval_cache/hf
        """
        raw = Path(cache_dir).expanduser().resolve()
        root = raw
        if raw.name == "hf":
            root = raw.parent
        hf = root / "hf"
        views = root / "views"
        collections = root / "collections"
        hf.mkdir(parents=True, exist_ok=True)
        views.mkdir(parents=True, exist_ok=True)
        collections.mkdir(parents=True, exist_ok=True)
        return cls(root=root, hf=hf, views=views, collections=collections)

