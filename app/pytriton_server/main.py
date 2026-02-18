from __future__ import annotations

import os

from app.model_assets import apply_model_env, resolve_gliner_model_source, resolve_token_classifier_model_source
from app.pytriton_server.registry import build_bindings


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def run() -> None:
    try:
        from pytriton.triton import Triton
    except Exception as exc:
        raise RuntimeError("PyTriton server runtime is not installed. Install with guardrails-service[cuda].") from exc

    enable_gliner = _env("GR_ENABLE_GLINER", "true").strip().lower() in {"1", "true", "yes", "on"}
    gliner_model_ref = _env("GR_PYTRITON_GLINER_MODEL_REF", "urchade/gliner_multi-v2.1")
    token_model_ref = _env("GR_PYTRITON_TOKEN_MODEL_REF", "scanpatch/pii-ner-nemotron")
    enable_nemotron = _env("GR_ENABLE_NEMOTRON", "false").strip().lower() in {"1", "true", "yes", "on"}
    model_dir = os.getenv("GR_MODEL_DIR")
    offline_mode = _env("GR_OFFLINE_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
    device = _env("GR_PYTRITON_DEVICE", "cuda")
    max_batch_size = int(_env("GR_PYTRITON_MAX_BATCH_SIZE", "32"))
    apply_model_env(model_dir=model_dir, offline_mode=offline_mode)
    gliner_source = ""
    if enable_gliner:
        gliner_source = resolve_gliner_model_source(
            model_name=gliner_model_ref,
            model_dir=model_dir,
            strict=offline_mode,
        )
    token_source = ""
    if enable_nemotron:
        token_source = resolve_token_classifier_model_source(
            model_name=token_model_ref,
            model_dir=model_dir,
            strict=offline_mode,
        )
    if not enable_gliner and not enable_nemotron:
        raise RuntimeError("no PyTriton models enabled (set GR_ENABLE_GLINER and/or GR_ENABLE_NEMOTRON)")

    bindings = build_bindings(
        gliner_model_ref=gliner_source,
        token_classifier_model_ref=token_source,
        device=device,
        max_batch_size=max_batch_size,
        enable_gliner=enable_gliner,
        enable_nemotron=enable_nemotron,
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
