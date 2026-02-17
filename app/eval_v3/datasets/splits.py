from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SamplerSpec:
    name: str  # none|random|label_balanced
    seed: int
    size: int | None = None
    min_per_label: int | None = None


@dataclass(frozen=True, slots=True)
class SubsetSpec:
    raw: str  # all|negatives|positives|language=ru|script_profile=...


@dataclass(frozen=True, slots=True)
class IndexSelection:
    indices: list[int]
    cache_path: Path
    from_cache: bool


def _dataset_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_").lower()


def _cache_path(
    *,
    cache_root: Path,
    dataset_id: str,
    split: str,
    subset: SubsetSpec,
    sampler: SamplerSpec,
    dataset_fingerprint: str | None,
) -> Path:
    signature = {
        "dataset": dataset_id,
        "split": split,
        "subset": subset.raw,
        "sampler": {"name": sampler.name, "seed": sampler.seed, "size": sampler.size, "min_per_label": sampler.min_per_label},
        "fingerprint": dataset_fingerprint or "",
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    split_dir = cache_root / "splits" / _dataset_slug(dataset_id)
    split_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [split, subset.raw, sampler.name, f"seed{sampler.seed}"]
    if sampler.size is not None:
        parts.append(f"size{sampler.size}")
    if sampler.min_per_label is not None:
        parts.append(f"min{sampler.min_per_label}")
    designator = "__".join(parts).strip().lower()
    designator = re.sub(r"[^a-z0-9._-]+", "_", designator).strip("_")
    if len(designator) > 120:
        designator = designator[:120].rstrip("_")
    return split_dir / f"{designator}__{key}.json"


def _apply_subset(
    *,
    subset: SubsetSpec,
    entity_counts: list[int | None],
    languages: list[str | None],
    script_profiles: list[str | None],
) -> list[int]:
    raw = subset.raw.strip()
    if raw in {"", "all"}:
        return list(range(len(entity_counts)))

    if raw == "negatives":
        if all(item is None for item in entity_counts):
            raise ValueError("subset=negatives requires entity_count field")
        return [idx for idx, val in enumerate(entity_counts) if val == 0]

    if raw == "positives":
        if all(item is None for item in entity_counts):
            raise ValueError("subset=positives requires entity_count field")
        return [idx for idx, val in enumerate(entity_counts) if val is not None and val > 0]

    if raw.startswith("language="):
        want = raw.split("=", 1)[1].strip().lower()
        if all(item is None for item in languages):
            raise ValueError("subset=language=... requires language field")
        return [idx for idx, lang in enumerate(languages) if (lang or "").strip().lower() == want]

    if raw.startswith("script_profile="):
        want = raw.split("=", 1)[1].strip()
        if all(item is None for item in script_profiles):
            raise ValueError("subset=script_profile=... requires script_profile field")
        return [idx for idx, val in enumerate(script_profiles) if (val or "").strip() == want]

    raise ValueError(f"unsupported subset spec: {subset.raw}")


def _sample_none(indices: list[int]) -> list[int]:
    return indices


def _sample_random(*, indices: list[int], size: int, seed: int) -> list[int]:
    if size <= 0:
        return []
    if len(indices) <= size:
        return sorted(indices)
    rng = random.Random(int(seed))
    out = list(indices)
    rng.shuffle(out)
    return sorted(out[:size])


def _sample_label_balanced(
    *,
    indices: list[int],
    sample_label_sets: list[set[str]],
    size: int,
    seed: int,
    min_per_label: int,
) -> list[int]:
    if size <= 0:
        return []
    if len(indices) <= size:
        return sorted(indices)

    rng = random.Random(int(seed))
    candidates_by_label: dict[str, list[int]] = {}
    for idx in indices:
        for label in sample_label_sets[idx]:
            candidates_by_label.setdefault(label, []).append(idx)

    for items in candidates_by_label.values():
        rng.shuffle(items)

    selected: set[int] = set()
    for label, candidates in sorted(candidates_by_label.items()):
        take = 0
        for idx in candidates:
            if idx in selected:
                continue
            selected.add(idx)
            take += 1
            if take >= min_per_label:
                break

    remaining = [idx for idx in indices if idx not in selected]
    rng.shuffle(remaining)
    need = max(0, size - len(selected))
    selected.update(remaining[:need])
    selected_list = sorted(selected)
    if len(selected_list) > size:
        rng.shuffle(selected_list)
        selected_list = sorted(selected_list[:size])
    return selected_list


def load_or_create_indices(
    *,
    cache_dir: str,
    dataset_id: str,
    split: str,
    subset: SubsetSpec,
    sampler: SamplerSpec,
    dataset_fingerprint: str | None,
    entity_counts: list[int | None],
    languages: list[str | None],
    script_profiles: list[str | None],
    sample_label_sets: list[set[str]],
) -> IndexSelection:
    cache_root = Path(cache_dir).expanduser().resolve() / "_eval_v3"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(
        cache_root=cache_root,
        dataset_id=dataset_id,
        split=split,
        subset=subset,
        sampler=sampler,
        dataset_fingerprint=dataset_fingerprint,
    )

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        indices = [int(item) for item in payload.get("indices", [])]
        return IndexSelection(indices=indices, cache_path=cache_path, from_cache=True)

    filtered = _apply_subset(
        subset=subset,
        entity_counts=entity_counts,
        languages=languages,
        script_profiles=script_profiles,
    )

    if sampler.name == "none":
        selected = _sample_none(filtered)
    elif sampler.name == "random":
        if sampler.size is None:
            raise ValueError("sampler=random requires size")
        selected = _sample_random(indices=filtered, size=int(sampler.size), seed=sampler.seed)
    elif sampler.name == "label_balanced":
        if sampler.size is None or sampler.min_per_label is None:
            raise ValueError("sampler=label_balanced requires size and min_per_label")
        selected = _sample_label_balanced(
            indices=filtered,
            sample_label_sets=sample_label_sets,
            size=int(sampler.size),
            seed=sampler.seed,
            min_per_label=int(sampler.min_per_label),
        )
    else:
        raise ValueError(f"unsupported sampler: {sampler.name}")

    payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "dataset_id": dataset_id,
        "split": split,
        "subset": subset.raw,
        "sampler": {
            "name": sampler.name,
            "seed": sampler.seed,
            "size": sampler.size,
            "min_per_label": sampler.min_per_label,
        },
        "dataset_fingerprint": dataset_fingerprint,
        "indices": selected,
        "filtered_count": len(filtered),
        "selected_count": len(selected),
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return IndexSelection(indices=selected, cache_path=cache_path, from_cache=False)
