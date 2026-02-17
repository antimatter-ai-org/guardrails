from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.eval.cache_paths import EvalCachePaths


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    collection: str
    dataset_ids: tuple[str, ...]
    source: str  # "cache" | "api"
    cache_path: Path


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_").lower() or "collection"


def resolve_collection(
    *,
    collection: str,
    cache_paths: EvalCachePaths,
    hf_token: str | None,
    refresh: bool,
) -> CollectionInfo:
    cache_path = cache_paths.collections / f"{_slugify(collection)}.json"
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        ids = tuple(str(x) for x in (payload.get("dataset_ids") or []))
        return CollectionInfo(collection=collection, dataset_ids=ids, source="cache", cache_path=cache_path)

    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required (project dependency).") from exc

    api = HfApi(token=hf_token)
    raw = api.get_collection(collection)
    items = getattr(raw, "items", None) or []

    dataset_ids: list[str] = []
    for item in items:
        item_type = getattr(item, "item_type", None) or getattr(item, "type", None) or ""
        item_id = getattr(item, "item_id", None) or getattr(item, "id", None) or ""
        # huggingface_hub models these as attributes; hf CLI returns JSON dicts.
        if isinstance(item, dict):
            item_type = item.get("item_type") or item.get("type") or item.get("itemType") or ""
            item_id = item.get("item_id") or item.get("id") or item.get("itemId") or ""

        if str(item_type).lower() != "dataset":
            continue
        if item_id:
            dataset_ids.append(str(item_id))

    dataset_ids = sorted(set(dataset_ids))
    payload: dict[str, Any] = {
        "collection": collection,
        "dataset_ids": dataset_ids,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return CollectionInfo(
        collection=collection,
        dataset_ids=tuple(dataset_ids),
        source="api",
        cache_path=cache_path,
    )

