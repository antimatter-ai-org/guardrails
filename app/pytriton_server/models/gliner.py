from __future__ import annotations

import json

import numpy as np

from app.pytriton_server.models.base import TritonModelBinding
from app.runtime.torch_runtime import resolve_torch_device


class GlinerTritonModel:
    def __init__(
        self,
        triton_model_name: str,
        gliner_hf_model_name: str,
        device: str,
        max_batch_size: int = 32,
    ) -> None:
        self._triton_model_name = triton_model_name
        self._gliner_hf_model_name = gliner_hf_model_name
        self._device = resolve_torch_device(device)
        self._max_batch_size = max_batch_size

        try:
            from gliner import GLiNER
        except Exception as exc:
            raise RuntimeError("gliner is not installed in pytriton server image") from exc

        self._model = GLiNER.from_pretrained(self._gliner_hf_model_name)
        if hasattr(self._model, "to"):
            self._model.to(self._device)

    @staticmethod
    def _decode_cell(cell: object) -> str:
        if isinstance(cell, (bytes, bytearray)):
            return cell.decode("utf-8")
        if isinstance(cell, str):
            return cell
        return str(cell)

    def _infer_impl(self, text: np.ndarray, labels_json: np.ndarray, threshold: np.ndarray) -> dict[str, np.ndarray]:
        batch_size = text.shape[0]
        outputs: list[list[bytes]] = []

        for idx in range(batch_size):
            raw_text = self._decode_cell(text[idx][0])
            raw_labels = self._decode_cell(labels_json[idx][0])
            raw_threshold = float(threshold[idx][0])

            labels = json.loads(raw_labels)
            if not isinstance(labels, list):
                labels = []
            labels = [str(item) for item in labels]

            predictions = self._model.predict_entities(raw_text, labels, threshold=raw_threshold)
            outputs.append([json.dumps(predictions, ensure_ascii=False).encode("utf-8")])

        return {"detections_json": np.array(outputs, dtype=object)}

    def binding(self) -> TritonModelBinding:
        from pytriton.decorators import batch
        from pytriton.model_config import ModelConfig, Tensor

        @batch
        def infer_func(**kwargs):
            return self._infer_impl(
                text=kwargs["text"],
                labels_json=kwargs["labels_json"],
                threshold=kwargs["threshold"],
            )

        return TritonModelBinding(
            name=self._triton_model_name,
            infer_func=infer_func,
            inputs=[
                Tensor(name="text", dtype=bytes, shape=(1,)),
                Tensor(name="labels_json", dtype=bytes, shape=(1,)),
                Tensor(name="threshold", dtype=np.float32, shape=(1,)),
            ],
            outputs=[
                Tensor(name="detections_json", dtype=bytes, shape=(1,)),
            ],
            config=ModelConfig(max_batch_size=self._max_batch_size),
        )
