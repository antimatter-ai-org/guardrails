from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

from app.eval_v3.config import load_eval_registry
from app.tools.rebalance_fast_splits import _default_fast_plans


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish dataset_info.json updates for suite datasets.")
    p.add_argument("--registry-path", default=str(Path("configs") / "eval" / "suites.yaml"))
    p.add_argument("--suite", default="guardrails_ru")
    p.add_argument("--cache-dir", default=".eval_cache/hf", help="HF cache dir (hub/datasets caches).")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--push", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--commit-message", default="Publish dataset_info.json (sync split sizes)")
    return p.parse_args()


def _configure_hf_cache(cache_dir: str) -> str:
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    # Do not override HF_HOME (can hide auth token); only point caches.
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_path / "datasets"))
    return str(cache_path)


def main() -> int:
    args = _parse_args()
    cache_dir = _configure_hf_cache(args.cache_dir)

    try:
        from datasets import load_dataset_builder  # type: ignore
        from datasets.splits import SplitInfo  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets package is required (install with guardrails-service[eval]).") from exc

    try:
        from huggingface_hub import CommitOperationAdd, HfApi, get_token  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub package is required (install with guardrails-service[eval]).") from exc

    token = get_token()
    if not token:
        raise RuntimeError("No HF auth token found. Run `hf auth login` or set HF_TOKEN.")
    os.environ.setdefault("HF_TOKEN", token)

    api = HfApi(token=token)
    reg = load_eval_registry(args.registry_path)
    plans = _default_fast_plans(args.registry_path, args.suite)

    for plan in plans:
        cfg = reg.datasets[plan.dataset_id]
        target_fast = int(plan.target_rows)
        print(f"\n== {plan.dataset_id} ==")
        print(f"hf_id: {cfg.hf_id}")
        print(f"set fast.num_examples -> {target_fast}")

        builder = load_dataset_builder(cfg.hf_id, token=True, cache_dir=cache_dir)
        info = builder.info
        if "fast" not in info.splits:
            raise RuntimeError(f"dataset has no 'fast' split in info: {cfg.hf_id}")
        orig = info.splits["fast"]
        info.splits["fast"] = SplitInfo(
            name="fast",
            num_bytes=int(getattr(orig, "num_bytes", 0) or 0),
            num_examples=target_fast,
            shard_lengths=getattr(orig, "shard_lengths", None),
            original_shard_lengths=getattr(orig, "original_shard_lengths", None),
            dataset_name=getattr(orig, "dataset_name", None),
        )

        with tempfile.TemporaryDirectory(prefix="gr_dataset_info_") as td:
            out_dir = Path(td)
            info.write_to_directory(str(out_dir))
            info_path = out_dir / "dataset_info.json"
            if not info_path.exists():
                raise RuntimeError(f"failed to write dataset_info.json for {cfg.hf_id}")

            # datasets expects dataset_infos.json (plural) on the Hub for split size verification.
            # It is a mapping: {config_name: dataset_info_dict}.
            infos_path = out_dir / "dataset_infos.json"
            payload = {str(info.config_name): __import__("json").loads(info_path.read_text(encoding="utf-8"))}
            infos_path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            if not bool(args.push) or bool(args.dry_run):
                print("dry_run: skipping push")
                continue

            ops: list[Any] = [
                CommitOperationAdd(path_in_repo="dataset_info.json", path_or_fileobj=str(info_path)),
                CommitOperationAdd(path_in_repo="dataset_infos.json", path_or_fileobj=str(infos_path)),
            ]
            api.create_commit(
                repo_id=cfg.hf_id,
                repo_type="dataset",
                operations=ops,
                commit_message=str(args.commit_message),
            )
            print("pushed dataset_info.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
