from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.config import load_policy_config
from app.model_assets import (
    apply_model_env,
    copy_natasha_models,
    gliner_local_dir,
)


def _collect_gliner_models(policy_path: str) -> list[str]:
    config = load_policy_config(policy_path)
    models: set[str] = set()
    for definition in config.detector_definitions.values():
        if definition.type.lower() != "gliner":
            continue
        model_name = str(definition.params.get("model_name", "urchade/gliner_multi-v2.1"))
        models.add(model_name)
    return sorted(models)


def _download_gliner_model(output_dir: str, model_name: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install project dependencies (for example: `uv sync --extra eval`)."
        ) from exc

    local_dir = gliner_local_dir(output_dir, model_name)
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
    )
    return str(local_dir)


def run(output_dir: str, policy_path: str, extra_gliner_models: list[str]) -> int:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    apply_model_env(model_dir=str(root), offline_mode=False)

    models = _collect_gliner_models(policy_path)
    models.extend(extra_gliner_models)
    gliner_models = sorted(set(models))

    downloaded: dict[str, str] = {}
    for model_name in gliner_models:
        local_path = _download_gliner_model(output_dir=str(root), model_name=model_name)
        downloaded[model_name] = local_path
        print(f"[ok] GLiNER model: {model_name} -> {local_path}")

    natasha_files = copy_natasha_models(output_dir=str(root))
    print(f"[ok] Natasha embedding -> {natasha_files['embedding']}")
    print(f"[ok] Natasha NER model -> {natasha_files['ner']}")

    manifest = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "policy_path": str(Path(policy_path).resolve()),
        "gliner_models": downloaded,
        "natasha": natasha_files,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifest -> {manifest_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all guardrails models into a single local directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to store models")
    parser.add_argument("--policy-path", default="configs/policy.yaml", help="Policy YAML used to discover GLiNER models")
    parser.add_argument(
        "--extra-gliner-model",
        action="append",
        default=[],
        help="Additional HuggingFace GLiNER repo id to pre-download (repeatable)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return run(
        output_dir=args.output_dir,
        policy_path=args.policy_path,
        extra_gliner_models=args.extra_gliner_model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
