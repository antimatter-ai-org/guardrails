from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.runtime.torch_runtime import resolve_torch_device


class GlinerRuntime(ABC):
    @abstractmethod
    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        raise NotImplementedError


class LocalTorchGlinerRuntime(GlinerRuntime):
    def __init__(
        self,
        model_name: str,
        preferred_device: str = "auto",
        use_fp16_on_cuda: bool = False,
    ) -> None:
        try:
            from gliner import GLiNER
        except Exception as exc:
            raise RuntimeError("gliner is not installed. Install with guardrails-service[ml].") from exc

        self.device = resolve_torch_device(preferred_device)
        self._model = GLiNER.from_pretrained(model_name)
        if hasattr(self._model, "to"):
            self._model.to(self.device)

        if use_fp16_on_cuda and self.device == "cuda" and hasattr(self._model, "half"):
            self._model.half()

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        return self._model.predict_entities(text, labels, threshold=threshold)


def build_gliner_runtime(
    backend: str,
    model_name: str,
    preferred_device: str = "auto",
    use_fp16_on_cuda: bool = False,
) -> GlinerRuntime:
    normalized = backend.strip().lower()

    # Future extension point: add triton/ray/kserve backends here.
    if normalized in {"local", "local_torch", "torch"}:
        return LocalTorchGlinerRuntime(
            model_name=model_name,
            preferred_device=preferred_device,
            use_fp16_on_cuda=use_fp16_on_cuda,
        )

    raise ValueError(
        f"unsupported GLiNER backend '{backend}'. Supported backends: local_torch"
    )
