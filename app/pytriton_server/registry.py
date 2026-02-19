from __future__ import annotations

from app.pytriton_server.models.base import TritonModelBinding
from app.pytriton_server.models.token_classifier import TokenClassifierTritonModel


def build_bindings(
    token_classifier_model_ref: str,
    device: str,
    max_batch_size: int,
    enable_nemotron: bool = True,
) -> list[TritonModelBinding]:
    models: list[object] = []
    if enable_nemotron:
        models.append(
            TokenClassifierTritonModel(
                triton_model_name="nemotron",
                hf_model_name=token_classifier_model_ref,
                device=device,
                aggregation_strategy="simple",
                max_batch_size=max_batch_size,
            )
        )
    return [model.binding() for model in models]
