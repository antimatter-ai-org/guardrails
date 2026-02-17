from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from app.eval_v3.config import load_eval_registry
from app.eval_v3.datasets.hf_span_dataset import load_hf_split, scan_hf_span_dataset


@dataclass(frozen=True, slots=True)
class FastSplitPlan:
    dataset_id: str
    hf_id: str
    target_rows: int
    negative_fraction: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebalance HF dataset fast splits for v3 eval.")
    p.add_argument("--registry-path", default=str(Path("configs") / "eval" / "suites.yaml"))
    p.add_argument("--suite", default="guardrails_ru")
    p.add_argument("--cache-dir", default=".eval_cache/hf", help="HF cache dir (HF_HOME/HF_DATASETS_CACHE root).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--push", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--commit-message", default=None)
    return p.parse_args()


def _configure_hf_cache(cache_dir: str) -> str:
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    # Do NOT set HF_HOME here: huggingface_hub resolves the auth token relative to HF_HOME.
    # Overriding HF_HOME can make an existing `hf auth login` token undiscoverable.
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_path / "datasets"))
    return str(cache_path)


def _ensure_deps() -> tuple[Any, Any, Any]:
    try:
        import datasets  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets package is required (install with guardrails-service[eval]).") from exc
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub package is required (install with guardrails-service[eval]).") from exc
    return datasets, (HfApi, get_token), (CommitOperationAdd, hf_hub_download)


def _default_fast_plans(registry_path: str, suite_name: str) -> list[FastSplitPlan]:
    """
    Target: <= ~10 minutes total eval runtime on a single GPU for the full suite.

    These values are tuned against observed throughput on H100 with v3 CUDA runtime + workers>1.
    """
    reg = load_eval_registry(registry_path)
    suite = reg.suites.get(suite_name)
    if suite is None:
        raise ValueError(f"unknown suite: {suite_name}")

    # Default plans by dataset id; any suite member not listed will be skipped explicitly.
    # Rationale: keep "fast" stable between machines by publishing the subset on HF itself.
    target_rows: dict[str, int] = {
        "antimatter-ai/guardrails-ru-russian-pii-66k-mvp-v1": 4000,
        "antimatter-ai/guardrails-ru-meddies-pii-cleaned-v1": 800,
        "antimatter-ai/guardrails-ru-scanpatch-pii-ner-controlled-v1": 1500,
        "antimatter-ai/guardrails-ru-rubai-ner-150k-personal-ru-v1": 2500,
        "antimatter-ai/guardrails-ru-kaggle-pii-data-detection-sample-v1": 800,
        "antimatter-ai/guardrails-ru-hf-pii-sample-en-v1": 800,
        "antimatter-ai/guardrails-ru-presidio-test-dataset-v1": 800,
    }

    plans: list[FastSplitPlan] = []
    for dataset_id in suite.datasets:
        cfg = reg.datasets.get(dataset_id)
        if cfg is None:
            continue
        if dataset_id not in target_rows:
            raise RuntimeError(f"no fast plan configured for suite dataset: {dataset_id}")
        plans.append(
            FastSplitPlan(
                dataset_id=dataset_id,
                hf_id=cfg.hf_id,
                target_rows=int(target_rows[dataset_id]),
                negative_fraction=0.20,
            )
        )
    return plans


def _pick_fast_indices(
    *,
    target_rows: int,
    negative_fraction: float,
    seed: int,
    entity_counts: list[int | None],
    label_sets: list[set[str]],
    scored_labels: set[str],
) -> tuple[list[int], dict[str, Any]]:
    if target_rows <= 0:
        return [], {"selected": 0}

    # Prefer entity_count if present; otherwise infer from label set.
    has_entity_count = not all(val is None for val in entity_counts)
    if has_entity_count:
        neg = [i for i, v in enumerate(entity_counts) if v == 0]
        pos = [i for i, v in enumerate(entity_counts) if v is not None and v > 0]
    else:
        neg = [i for i, s in enumerate(label_sets) if not s]
        pos = [i for i, s in enumerate(label_sets) if s]

    total = len(entity_counts)
    target_rows = min(int(target_rows), total)

    rng = Random(int(seed))

    want_neg = int(round(float(target_rows) * float(negative_fraction)))
    want_neg = min(want_neg, len(neg))
    want_pos = target_rows - want_neg
    want_pos = min(want_pos, len(pos))
    # If we couldn't fill positives, shift budget back to negatives (if available).
    want_neg = min(target_rows - want_pos, len(neg))

    scored_labels_sorted = sorted({s for s in scored_labels if s})
    scored_label_sets: list[set[str]] = [labels & set(scored_labels_sorted) for labels in label_sets]

    # Determine a conservative min-per-label that doesn't overwhelm small splits.
    denom = max(1, 2 * len(scored_labels_sorted))
    min_per_label = max(10, min(30, want_pos // denom)) if want_pos > 0 else 0

    # Label-balanced sampling over positives.
    candidates_by_label: dict[str, list[int]] = {label: [] for label in scored_labels_sorted}
    for idx in pos:
        for label in scored_label_sets[idx]:
            if label in candidates_by_label:
                candidates_by_label[label].append(idx)
    for items in candidates_by_label.values():
        rng.shuffle(items)

    selected_pos: set[int] = set()
    for label in scored_labels_sorted:
        take = 0
        for idx in candidates_by_label.get(label, []):
            if idx in selected_pos:
                continue
            selected_pos.add(idx)
            take += 1
            if take >= int(min_per_label):
                break

    remaining_pos = [idx for idx in pos if idx not in selected_pos]
    rng.shuffle(remaining_pos)
    need_pos = max(0, want_pos - len(selected_pos))
    selected_pos.update(remaining_pos[:need_pos])

    selected_neg: set[int] = set()
    if want_neg > 0 and neg:
        neg_shuffled = list(neg)
        rng.shuffle(neg_shuffled)
        selected_neg.update(neg_shuffled[:want_neg])

    selected = sorted(selected_pos | selected_neg)
    meta = {
        "total_rows": total,
        "target_rows": target_rows,
        "selected_rows": len(selected),
        "want_pos": want_pos,
        "want_neg": want_neg,
        "selected_pos": len(selected_pos),
        "selected_neg": len(selected_neg),
        "min_per_label": int(min_per_label),
        "scored_labels": scored_labels_sorted,
    }
    return selected, meta


def _count_fast_labels(*, ds_fast: Any, spans_field: str, label_map: dict[str, str]) -> tuple[int, dict[str, int]]:
    # Approximate: count entity spans by canonical label.
    from app.eval_v3.datasets.hf_span_dataset import _map_gold_label  # local import to avoid making it public

    neg_rows = 0
    counts: Counter[str] = Counter()
    for row in ds_fast:
        spans = (row or {}).get(spans_field) or []
        if not spans:
            neg_rows += 1
        if isinstance(spans, list):
            for item in spans:
                if not isinstance(item, dict):
                    continue
                canonical = _map_gold_label(item.get("label"), label_map)
                if canonical:
                    counts[str(canonical)] += 1
    return neg_rows, dict(sorted(counts.items()))


def _update_derivation_stats(
    *,
    stats: dict[str, Any],
    fast_rows: int,
    negative_rows_fast: int,
    label_counts_fast: dict[str, int],
    seed: int,
    negative_fraction: float,
) -> dict[str, Any]:
    out = dict(stats)
    # Common keys seen across our datasets.
    if "fast_rows" in out:
        out["fast_rows"] = int(fast_rows)
    if "negative_rows_fast" in out:
        out["negative_rows_fast"] = int(negative_rows_fast)
    if "label_counts_fast" in out and isinstance(out.get("label_counts_fast"), dict):
        out["label_counts_fast"] = dict(label_counts_fast)
    if "core_label_counts_fast" in out and isinstance(out.get("core_label_counts_fast"), dict):
        out["core_label_counts_fast"] = dict(label_counts_fast)

    # Add a non-breaking provenance block.
    out["fast_rebalanced_v3"] = {
        "seed": int(seed),
        "negative_fraction": float(negative_fraction),
        "fast_rows": int(fast_rows),
        "negative_rows_fast": int(negative_rows_fast),
    }
    return out


def main() -> int:
    args = _parse_args()
    datasets, (HfApi, get_token), (CommitOperationAdd, hf_hub_download) = _ensure_deps()

    token = get_token()
    if not token:
        raise RuntimeError("No HF auth token found. Run `hf auth login` or set HF_TOKEN.")
    # Pin token in env so subsequent cache env tweaks can't hide it.
    os.environ.setdefault("HF_TOKEN", token)

    cache_dir = _configure_hf_cache(args.cache_dir)

    plans = _default_fast_plans(args.registry_path, args.suite)
    reg = load_eval_registry(args.registry_path)

    api = HfApi(token=token)

    for plan in plans:
        cfg = reg.datasets[plan.dataset_id]
        print(f"\n== {plan.dataset_id} ==")
        print(f"hf_id: {plan.hf_id}")

        ds_full, fingerprint = load_hf_split(
            hf_id=plan.hf_id,
            split="full",
            cache_dir=cache_dir,
            hf_token=True,
        )
        entity_counts, _, _, label_sets = scan_hf_span_dataset(ds=ds_full, spans_field=cfg.spans_field, label_map=cfg.label_map)

        indices, meta = _pick_fast_indices(
            target_rows=plan.target_rows,
            negative_fraction=plan.negative_fraction,
            seed=int(args.seed),
            entity_counts=entity_counts,
            label_sets=label_sets,
            scored_labels=set(cfg.scored_labels),
        )
        print("plan:", json.dumps(meta, ensure_ascii=False))

        ds_fast = ds_full.select(indices)
        negative_rows_fast, label_counts_fast = _count_fast_labels(ds_fast=ds_fast, spans_field=cfg.spans_field, label_map=cfg.label_map)
        print("fast_stats:", json.dumps({"negative_rows_fast": negative_rows_fast, "label_counts_fast": label_counts_fast}, ensure_ascii=False))

        if not bool(args.push) or bool(args.dry_run):
            print("dry_run: skipping push")
            continue

        repo_files = set(api.list_repo_files(repo_id=plan.hf_id, repo_type="dataset"))
        stats_paths = [p for p in ("derivation_stats.json", "data/derivation_stats.json") if p in repo_files]

        # Materialize parquet locally and create a single commit replacing fast parquet (and stats if present).
        with tempfile.TemporaryDirectory(prefix="gr_fast_rebalance_") as td:
            td_path = Path(td)
            parquet_path = td_path / "fast-00000-of-00001.parquet"
            ds_fast.to_parquet(str(parquet_path))

            ops: list[Any] = [
                CommitOperationAdd(path_in_repo="data/fast-00000-of-00001.parquet", path_or_fileobj=str(parquet_path)),
            ]

            # Update README.md front matter dataset_info.splits[fast].num_examples so `datasets.load_dataset`
            # doesn't enforce stale expected split sizes.
            try:
                readme_src = hf_hub_download(repo_id=plan.hf_id, repo_type="dataset", filename="README.md")
                readme_text = Path(readme_src).read_text(encoding="utf-8")
                if readme_text.startswith("---"):
                    end = readme_text.find("\n---", 3)
                    if end > 0:
                        fm_raw = readme_text[3:end].lstrip("\n")
                        body = readme_text[end + 4 :].lstrip("\n")
                        import yaml  # type: ignore

                        meta = yaml.safe_load(fm_raw) or {}
                        di = meta.get("dataset_info") or {}
                        splits = di.get("splits") or []
                        if isinstance(splits, list):
                            for item in splits:
                                if isinstance(item, dict) and item.get("name") == "fast":
                                    item["num_examples"] = int(len(ds_fast))
                        meta["dataset_info"] = di
                        fm_new = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
                        readme_out = td_path / "README.md"
                        readme_out.write_text(f"---\n{fm_new}\n---\n\n{body}", encoding="utf-8")
                        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme_out)))
            except Exception:
                pass

            # Publish dataset_info.json with updated split sizes so `datasets.load_dataset(...)`
            # doesn't fail split verification on machines with cached/stale expected sizes.
            try:
                builder = datasets.load_dataset_builder(plan.hf_id, token=True, cache_dir=cache_dir)
                info = builder.info
                # Keep the existing split metadata but update fast num_examples to the new split size.
                from datasets.splits import SplitInfo  # type: ignore

                if "fast" in info.splits:
                    orig = info.splits["fast"]
                    info.splits["fast"] = SplitInfo(
                        name="fast",
                        num_bytes=int(getattr(orig, "num_bytes", 0) or 0),
                        num_examples=int(len(ds_fast)),
                        shard_lengths=getattr(orig, "shard_lengths", None),
                        original_shard_lengths=getattr(orig, "original_shard_lengths", None),
                        dataset_name=getattr(orig, "dataset_name", None),
                    )
                info_dir = td_path / "_dataset_info"
                info_dir.mkdir(parents=True, exist_ok=True)
                info.write_to_directory(str(info_dir))
                info_path = info_dir / "dataset_info.json"
                if info_path.exists():
                    ops.append(CommitOperationAdd(path_in_repo="dataset_info.json", path_or_fileobj=str(info_path)))
                    # Also publish dataset_infos.json (plural) for completeness.
                    infos_path = info_dir / "dataset_infos.json"
                    payload = {str(info.config_name): __import__("json").loads(info_path.read_text(encoding="utf-8"))}
                    infos_path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    ops.append(CommitOperationAdd(path_in_repo="dataset_infos.json", path_or_fileobj=str(infos_path)))
            except Exception:
                # Best effort: fast parquet is still updated; dataset-info mismatch can be resolved by
                # clearing local caches. Prefer not to fail the entire publish step.
                pass

            for stats_path in stats_paths:
                local_stats = hf_hub_download(repo_id=plan.hf_id, repo_type="dataset", filename=stats_path)
                payload = json.loads(Path(local_stats).read_text(encoding="utf-8"))
                updated = _update_derivation_stats(
                    stats=payload,
                    fast_rows=len(ds_fast),
                    negative_rows_fast=negative_rows_fast,
                    label_counts_fast=label_counts_fast,
                    seed=int(args.seed),
                    negative_fraction=float(plan.negative_fraction),
                )
                out_stats = td_path / Path(stats_path).name
                out_stats.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
                ops.append(CommitOperationAdd(path_in_repo=stats_path, path_or_fileobj=str(out_stats)))

            msg = args.commit_message or (
                f"Rebalance fast split for v3 eval (rows={len(ds_fast)}, seed={int(args.seed)}, neg_frac={plan.negative_fraction})"
            )
            api.create_commit(
                repo_id=plan.hf_id,
                repo_type="dataset",
                operations=ops,
                commit_message=msg,
            )
            print(f"pushed: {plan.hf_id} fast_rows={len(ds_fast)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
