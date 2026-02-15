from __future__ import annotations

from app.pytriton_server.models.base import TritonModelBinding
from app.pytriton_server.models.gliner import GlinerTritonModel
from app.pytriton_server.models.token_classifier import TokenClassifierTritonModel


def build_bindings(
    gliner_model_ref: str,
    token_classifier_model_ref: str,
    device: str,
    max_batch_size: int,
) -> list[TritonModelBinding]:
    models = [
        GlinerTritonModel(
            triton_model_name="gliner",
            gliner_hf_model_name=gliner_model_ref,
            device=device,
            max_batch_size=max_batch_size,
        ),
        TokenClassifierTritonModel(
            triton_model_name="nemotron",
            hf_model_name=token_classifier_model_ref,
            device=device,
            aggregation_strategy="simple",
            max_batch_size=max_batch_size,
        ),
    ]
    return [model.binding() for model in models]
