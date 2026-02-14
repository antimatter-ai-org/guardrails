from __future__ import annotations

import os

from app.model_assets import apply_model_env, resolve_gliner_model_source
from app.pytriton_server.registry import build_bindings


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def run() -> None:
    try:
        from pytriton.triton import Triton
    except Exception as exc:
        raise RuntimeError("PyTriton server runtime is not installed. Install with guardrails-service[cuda].") from exc

    triton_model_name = _env("GR_PYTRITON_MODEL_NAME", "gliner")
    model_ref = _env("GR_PYTRITON_MODEL_REF", "urchade/gliner_multi-v2.1")
    model_dir = os.getenv("GR_MODEL_DIR")
    offline_mode = _env("GR_OFFLINE_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
    device = _env("GR_PYTRITON_DEVICE", "cuda")
    max_batch_size = int(_env("GR_PYTRITON_MAX_BATCH_SIZE", "32"))
    apply_model_env(model_dir=model_dir, offline_mode=offline_mode)
    gliner_source = resolve_gliner_model_source(
        model_name=model_ref,
        model_dir=model_dir,
        strict=offline_mode,
    )

    bindings = build_bindings(
        triton_model_name=triton_model_name,
        model_ref=gliner_source,
        device=device,
        max_batch_size=max_batch_size,
    )

    with Triton() as triton:
        for binding in bindings:
            triton.bind(
                model_name=binding.name,
                infer_func=binding.infer_func,
                inputs=binding.inputs,
                outputs=binding.outputs,
                config=binding.config,
            )
        triton.serve()


if __name__ == "__main__":
    run()
