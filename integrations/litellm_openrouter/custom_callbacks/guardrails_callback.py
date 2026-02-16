import copy
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable
from uuid import uuid4

import httpx
from fastapi import HTTPException

try:
    from litellm.integrations.custom_logger import CustomLogger
except Exception:  # pragma: no cover - used only in router runtime
    class CustomLogger:  # type: ignore[no-redef]
        pass

try:
    from litellm.types.utils import ModelResponseStream as _ModelResponseStream
except Exception:  # pragma: no cover - used only when litellm is absent in unit tests
    _ModelResponseStream = None

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _TextRef:
    item_id: str
    text: str
    setter: Callable[[str], None]


@dataclass(slots=True)
class _StreamTextRef:
    choice_index: int
    text: str
    setter: Callable[[str], None]


def _safe_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _safe_set(value: Any, key: str, new_value: Any) -> None:
    if isinstance(value, dict):
        value[key] = new_value
        return
    setattr(value, key, new_value)


class GuardrailsCallback(CustomLogger):
    def __init__(self) -> None:
        self.guardrails_url = os.getenv("GUARDRAILS_URL", "http://guardrails:8080").rstrip("/")
        self.guardrails_policy_id = os.getenv("GUARDRAILS_POLICY_ID", "external_default")
        self.guardrails_timeout_s = float(os.getenv("GUARDRAILS_TIMEOUT_S", "15"))
        self.guardrails_output_scope = os.getenv("GUARDRAILS_OUTPUT_SCOPE", "FULL")

    @staticmethod
    def _ensure_metadata(data: dict[str, Any]) -> dict[str, Any]:
        metadata_key = "metadata"
        if isinstance(data.get("litellm_metadata"), dict):
            metadata_key = "litellm_metadata"
        metadata = data.get(metadata_key)
        if not isinstance(metadata, dict):
            metadata = {}
            data[metadata_key] = metadata
        return metadata

    @classmethod
    def _read_ctx(cls, data: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(data, dict):
            return None
        metadata = cls._ensure_metadata(data)
        ctx = metadata.get("_guardrails_ctx")
        if isinstance(ctx, dict):
            return ctx
        return None

    @classmethod
    def _write_ctx(cls, data: dict[str, Any], ctx: dict[str, Any]) -> None:
        metadata = cls._ensure_metadata(data)
        metadata["_guardrails_ctx"] = ctx

    @classmethod
    def _clear_ctx(cls, data: dict[str, Any]) -> None:
        metadata = cls._ensure_metadata(data)
        metadata.pop("_guardrails_ctx", None)

    @staticmethod
    def _new_request_id() -> str:
        return f"litellm-gr-{uuid4().hex}"

    @staticmethod
    def _is_text_part(part: Any) -> bool:
        if not isinstance(part, dict):
            return False
        part_type = str(part.get("type", "")).strip().lower()
        if not part_type:
            return isinstance(part.get("text"), str)
        return "text" in part_type

    def _collect_input_text_refs(self, data: dict[str, Any]) -> list[_TextRef]:
        refs: list[_TextRef] = []
        text_idx = 0
        messages = data.get("messages")
        if isinstance(messages, list):
            for msg_idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if isinstance(content, str):
                    item_id = f"msg-{msg_idx}"
                    refs.append(
                        _TextRef(
                            item_id=item_id,
                            text=content,
                            setter=lambda value, message=msg: message.__setitem__("content", value),
                        )
                    )
                    continue
                if isinstance(content, list):
                    for part_idx, part in enumerate(content):
                        if not self._is_text_part(part):
                            continue
                        part_text = part.get("text")
                        if not isinstance(part_text, str):
                            continue
                        item_id = f"msg-{msg_idx}-part-{part_idx}"
                        refs.append(
                            _TextRef(
                                item_id=item_id,
                                text=part_text,
                                setter=lambda value, message_part=part: message_part.__setitem__("text", value),
                            )
                        )

        prompt = data.get("prompt")
        if isinstance(prompt, str):
            refs.append(
                _TextRef(
                    item_id=f"prompt-{text_idx}",
                    text=prompt,
                    setter=lambda value: data.__setitem__("prompt", value),
                )
            )
            text_idx += 1

        direct_input = data.get("input")
        if isinstance(direct_input, str):
            refs.append(
                _TextRef(
                    item_id=f"input-{text_idx}",
                    text=direct_input,
                    setter=lambda value: data.__setitem__("input", value),
                )
            )
        return refs

    def _collect_output_text_refs(self, response: Any) -> list[_TextRef]:
        refs: list[_TextRef] = []
        choices = _safe_get(response, "choices", [])
        if not isinstance(choices, list):
            return refs

        for choice_idx, choice in enumerate(choices):
            message = _safe_get(choice, "message")
            if message is None:
                continue
            content = _safe_get(message, "content")
            if isinstance(content, str):
                refs.append(
                    _TextRef(
                        item_id=f"choice-{choice_idx}",
                        text=content,
                        setter=lambda value, message_obj=message: _safe_set(message_obj, "content", value),
                    )
                )
                continue
            if isinstance(content, list):
                for part_idx, part in enumerate(content):
                    if not isinstance(part, dict):
                        continue
                    part_text = part.get("text")
                    if not isinstance(part_text, str):
                        continue
                    refs.append(
                        _TextRef(
                            item_id=f"choice-{choice_idx}-part-{part_idx}",
                            text=part_text,
                            setter=lambda value, message_part=part: message_part.__setitem__("text", value),
                        )
                    )
        return refs

    def _collect_stream_delta_refs(self, chunk: Any) -> tuple[list[_StreamTextRef], set[int]]:
        refs: list[_StreamTextRef] = []
        completed_choices: set[int] = set()
        choices = _safe_get(chunk, "choices", [])
        if not isinstance(choices, list):
            return refs, completed_choices

        for idx, choice in enumerate(choices):
            choice_index = _safe_get(choice, "index", idx)
            try:
                numeric_choice_index = int(choice_index)
            except Exception:
                numeric_choice_index = idx

            finish_reason = _safe_get(choice, "finish_reason")
            if finish_reason is not None:
                completed_choices.add(numeric_choice_index)

            delta = _safe_get(choice, "delta")
            if delta is None:
                continue
            content = _safe_get(delta, "content")
            if isinstance(content, str) and content:
                refs.append(
                    _StreamTextRef(
                        choice_index=numeric_choice_index,
                        text=content,
                        setter=lambda value, delta_obj=delta: _safe_set(delta_obj, "content", value),
                    )
                )

        return refs, completed_choices

    @staticmethod
    def _choice_stream_id(choice_index: int) -> str:
        return f"choice-{choice_index}"

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.guardrails_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.guardrails_timeout_s) as client:
                response = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=503, detail=f"guardrails timeout: {exc}") from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"guardrails unavailable: {exc}") from exc

        if response.status_code >= 400:
            detail: str = response.text
            try:
                body = response.json()
                if isinstance(body, dict) and "detail" in body:
                    detail = str(body["detail"])
            except Exception:
                pass
            raise HTTPException(status_code=503, detail=f"guardrails error [{response.status_code}]: {detail}")

        try:
            parsed = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=503, detail="guardrails returned non-json response") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=503, detail="guardrails returned invalid response type")
        return parsed

    async def _guardrails_apply(
        self,
        *,
        source: str,
        mode: str,
        content: list[dict[str, str]],
        request_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        transform: dict[str, Any] = {
            "type": "reversible_mask",
            "mode": mode,
            "session": {},
        }
        if session_id is not None:
            transform["session"] = {"id": session_id}
        payload = {
            "request_id": request_id,
            "policy_id": self.guardrails_policy_id,
            "source": source,
            "content": content,
            "transforms": [transform],
            "output_scope": self.guardrails_output_scope,
            "trace": "NONE",
        }
        return await self._post_json("/v1/guardrails/apply", payload)

    async def _guardrails_apply_stream(
        self,
        *,
        session_id: str,
        stream_id: str,
        chunk: str,
        final: bool,
        request_id: str,
    ) -> dict[str, Any]:
        payload = {
            "request_id": request_id,
            "policy_id": self.guardrails_policy_id,
            "source": "OUTPUT",
            "transforms": [
                {
                    "type": "reversible_mask",
                    "mode": "REIDENTIFY",
                    "session": {"id": session_id},
                }
            ],
            "output_scope": self.guardrails_output_scope,
            "trace": "NONE",
            "stream": {
                "id": stream_id,
                "chunk": chunk,
                "final": final,
            },
        }
        return await self._post_json("/v1/guardrails/apply-stream", payload)

    async def _guardrails_finalize(self, session_id: str) -> None:
        await self._post_json(f"/v1/guardrails/sessions/{session_id}/finalize", {})

    async def _finalize_best_effort(self, session_id: str) -> None:
        try:
            await self._guardrails_finalize(session_id)
        except Exception as exc:
            logger.warning("guardrails finalize failed for session=%s: %s", session_id, exc)

    @staticmethod
    def _build_synthetic_chunk(template_chunk: Any, choice_index: int, content: str) -> Any:
        if isinstance(template_chunk, dict):
            synthetic = copy.deepcopy(template_chunk)
            synthetic["choices"] = [
                {
                    "index": choice_index,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ]
            return synthetic

        if _ModelResponseStream is not None:
            return _ModelResponseStream(
                id=_safe_get(template_chunk, "id", None),
                created=_safe_get(template_chunk, "created", None),
                model=_safe_get(template_chunk, "model", None),
                choices=[
                    {
                        "index": choice_index,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            )
        return {
            "id": _safe_get(template_chunk, "id", "chatcmpl-synthetic"),
            "object": "chat.completion.chunk",
            "created": _safe_get(template_chunk, "created", 0),
            "model": _safe_get(template_chunk, "model", None),
            "choices": [
                {
                    "index": choice_index,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        }

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        text_refs = self._collect_input_text_refs(data)
        if not text_refs:
            return data

        request_id = self._new_request_id()
        response = await self._guardrails_apply(
            source="INPUT",
            mode="DEIDENTIFY",
            request_id=request_id,
            content=[{"id": ref.item_id, "text": ref.text} for ref in text_refs],
        )
        outputs = response.get("outputs", [])
        if not isinstance(outputs, list) or len(outputs) != len(text_refs):
            raise HTTPException(status_code=503, detail="guardrails invalid apply response: outputs mismatch")

        for ref, output_item in zip(text_refs, outputs, strict=True):
            masked_text = output_item.get("text", "") if isinstance(output_item, dict) else ""
            if not isinstance(masked_text, str):
                raise HTTPException(status_code=503, detail="guardrails invalid apply response: output text must be string")
            ref.setter(masked_text)

        session = response.get("session", {})
        session_id = session.get("id") if isinstance(session, dict) else None
        if not isinstance(session_id, str) or not session_id:
            raise HTTPException(status_code=503, detail="guardrails did not return session id")

        self._write_ctx(
            data,
            {
                "session_id": session_id,
                "policy_id": response.get("policy_id", self.guardrails_policy_id),
                "request_id": request_id,
            },
        )
        return data

    async def async_post_call_success_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        ctx = self._read_ctx(data)
        if not ctx:
            return response

        session_id = ctx.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise HTTPException(status_code=503, detail="guardrails session context is invalid")

        try:
            text_refs = self._collect_output_text_refs(response)
            if text_refs:
                apply_response = await self._guardrails_apply(
                    source="OUTPUT",
                    mode="REIDENTIFY",
                    request_id=str(ctx.get("request_id") or self._new_request_id()),
                    session_id=session_id,
                    content=[{"id": ref.item_id, "text": ref.text} for ref in text_refs],
                )
                outputs = apply_response.get("outputs", [])
                if not isinstance(outputs, list) or len(outputs) != len(text_refs):
                    raise HTTPException(status_code=503, detail="guardrails invalid reidentify response: outputs mismatch")
                for ref, output_item in zip(text_refs, outputs, strict=True):
                    unmasked_text = output_item.get("text", "") if isinstance(output_item, dict) else ""
                    if not isinstance(unmasked_text, str):
                        raise HTTPException(
                            status_code=503,
                            detail="guardrails invalid reidentify response: output text must be string",
                        )
                    ref.setter(unmasked_text)
        except Exception:
            await self._finalize_best_effort(session_id)
            self._clear_ctx(data)
            raise

        await self._finalize_best_effort(session_id)
        self._clear_ctx(data)
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        ctx = self._read_ctx(request_data)
        if not ctx:
            async for chunk in response:
                yield chunk
            return

        session_id = ctx.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise HTTPException(status_code=503, detail="guardrails session context is invalid")

        request_id = str(ctx.get("request_id") or self._new_request_id())
        active_stream_ids: set[str] = set()
        last_chunk_per_choice: dict[int, Any] = {}

        try:
            async for chunk in response:
                working_chunk = chunk
                if hasattr(chunk, "model_dump"):
                    try:
                        working_chunk = chunk.model_dump()  # type: ignore[assignment]
                    except Exception:
                        working_chunk = chunk
                elif isinstance(chunk, dict):
                    working_chunk = copy.deepcopy(chunk)

                refs, completed_choice_indexes = self._collect_stream_delta_refs(working_chunk)
                for ref in refs:
                    stream_id = self._choice_stream_id(ref.choice_index)
                    active_stream_ids.add(stream_id)
                    apply_response = await self._guardrails_apply_stream(
                        session_id=session_id,
                        stream_id=stream_id,
                        chunk=ref.text,
                        final=False,
                        request_id=request_id,
                    )
                    output_chunk = apply_response.get("output_chunk", "")
                    if not isinstance(output_chunk, str):
                        raise HTTPException(status_code=503, detail="guardrails invalid stream response: output_chunk must be string")
                    ref.setter(output_chunk)
                    last_chunk_per_choice[ref.choice_index] = working_chunk

                for choice_index in completed_choice_indexes:
                    last_chunk_per_choice.setdefault(choice_index, working_chunk)

                emit_chunk = working_chunk
                if _ModelResponseStream is not None and isinstance(working_chunk, dict):
                    try:
                        emit_chunk = _ModelResponseStream(**working_chunk)
                    except Exception:
                        emit_chunk = working_chunk
                yield emit_chunk
        except Exception:
            await self._finalize_best_effort(session_id)
            self._clear_ctx(request_data)
            raise

        for stream_id in sorted(active_stream_ids):
            try:
                choice_index = int(stream_id.removeprefix("choice-"))
            except Exception:
                choice_index = 0
            flush_response = await self._guardrails_apply_stream(
                session_id=session_id,
                stream_id=stream_id,
                chunk="",
                final=True,
                request_id=request_id,
            )
            output_chunk = flush_response.get("output_chunk", "")
            if not isinstance(output_chunk, str):
                raise HTTPException(status_code=503, detail="guardrails invalid stream flush response: output_chunk must be string")
            if output_chunk:
                template_chunk = last_chunk_per_choice.get(choice_index, {})
                yield self._build_synthetic_chunk(
                    template_chunk=template_chunk,
                    choice_index=choice_index,
                    content=output_chunk,
                )

        await self._finalize_best_effort(session_id)
        self._clear_ctx(request_data)

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, Any],
        original_exception: Exception,
        user_api_key_dict: Any,
        traceback_str: str | None = None,
    ) -> HTTPException | None:
        ctx = self._read_ctx(request_data)
        if not ctx:
            return None
        session_id = ctx.get("session_id")
        if isinstance(session_id, str) and session_id:
            await self._finalize_best_effort(session_id)
        self._clear_ctx(request_data)
        if isinstance(original_exception, HTTPException):
            return original_exception
        return None


guardrails_callback = GuardrailsCallback()
