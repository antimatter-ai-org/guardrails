from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.eval_v3.config import load_eval_registry


def _configure_hf_cache(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(root / "transformers")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-download eval datasets for offline/air-gapped operation.")
    p.add_argument("--output-dir", required=True, help="Directory to store dataset cache + manifest.")
    p.add_argument("--registry-path", default=str(Path("configs") / "eval" / "suites.yaml"))
    p.add_argument("--suite", default="guardrails_ru")
    p.add_argument("--splits", default="fast,full", help="Comma-separated list of splits to download (e.g. fast,full).")
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    return p.parse_args()


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _best_effort_dataset_sha(dataset_id: str, hf_token: str | bool | None) -> str | None:
    try:
        from huggingface_hub import HfApi  # type: ignore
    except Exception:
        return None
    try:
        info = HfApi().dataset_info(repo_id=dataset_id, token=hf_token)
        return str(getattr(info, "sha", None) or "") or None
    except Exception:
        return None


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def main() -> int:
    args = _parse_args()
    registry = load_eval_registry(args.registry_path)
    suite = registry.suites.get(args.suite)
    if suite is None:
        raise RuntimeError(f"unknown suite: {args.suite}")

    splits = _split_list(args.splits)
    if not splits:
        raise RuntimeError("--splits must be non-empty")

    hf_token_env = os.getenv(args.hf_token_env)
    hf_token: str | bool | None = hf_token_env if hf_token_env else True
    out_dir = Path(args.output_dir).expanduser().resolve()
    cache_root = out_dir / "hf_cache"
    _configure_hf_cache(cache_root)

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets package is required. Install with guardrails-service[eval].") from exc

    downloaded: list[dict[str, Any]] = []
    for dataset_id in suite.datasets:
        cfg = registry.datasets[dataset_id]
        sha = _best_effort_dataset_sha(cfg.hf_id, hf_token)
        for split in splits:
            print(f"[download] dataset={cfg.hf_id} split={split}", flush=True)
            ds = load_dataset(cfg.hf_id, split=split, token=hf_token, cache_dir=str(cache_root))
            # Touch a few rows to ensure files are actually materialized.
            _ = ds[0] if len(ds) > 0 else None
            downloaded.append(
                {
                    "dataset_id": dataset_id,
                    "hf_id": cfg.hf_id,
                    "split": split,
                    "sha": sha,
                    "fingerprint": str(getattr(ds, "_fingerprint", "")) or None,
                    "rows": int(len(ds)),
                }
            )

    manifest = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "registry_path": str(Path(args.registry_path).resolve()),
        "suite": suite.name,
        "splits": splits,
        "datasets": downloaded,
        "cache_root": str(cache_root),
        "cache_size_bytes": _dir_size_bytes(cache_root),
    }
    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifest -> {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
