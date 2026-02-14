from __future__ import annotations

from app.pytriton_server.models.base import TritonModelBinding
from app.pytriton_server.models.gliner import GlinerTritonModel


def build_bindings(
    triton_model_name: str,
    model_ref: str,
    device: str,
    max_batch_size: int,
) -> list[TritonModelBinding]:
    models = [
        GlinerTritonModel(
            triton_model_name=triton_model_name,
            gliner_hf_model_name=model_ref,
            device=device,
            max_batch_size=max_batch_size,
        )
    ]
    return [model.binding() for model in models]
