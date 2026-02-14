from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TextSlot:
    path: tuple[Any, ...]
    text: str


def copy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(payload)


def _iter_message_content(payload: dict[str, Any]) -> list[TextSlot]:
    slots: list[TextSlot] = []
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return slots

    for msg_idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if isinstance(content, str):
            slots.append(TextSlot(path=("messages", msg_idx, "content"), text=content))
        elif isinstance(content, list):
            for part_idx, part in enumerate(content):
                if not isinstance(part, dict):
                    continue
                if part.get("type", "text") == "text" and isinstance(part.get("text"), str):
                    slots.append(TextSlot(path=("messages", msg_idx, "content", part_idx, "text"), text=part["text"]))

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call_idx, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    continue
                function_payload = call.get("function")
                if not isinstance(function_payload, dict):
                    continue
                arguments = function_payload.get("arguments")
                if isinstance(arguments, str):
                    slots.append(
                        TextSlot(
                            path=("messages", msg_idx, "tool_calls", call_idx, "function", "arguments"),
                            text=arguments,
                        )
                    )

    return slots


def collect_request_text_slots(payload: dict[str, Any]) -> list[TextSlot]:
    return _iter_message_content(payload)


def collect_response_text_slots(payload: dict[str, Any]) -> list[TextSlot]:
    slots: list[TextSlot] = []
    choices = payload.get("choices", [])
    if not isinstance(choices, list):
        return slots

    for choice_idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                slots.append(TextSlot(path=("choices", choice_idx, "message", "content"), text=content))
            elif isinstance(content, list):
                for part_idx, part in enumerate(content):
                    if isinstance(part, dict) and part.get("type", "text") == "text" and isinstance(part.get("text"), str):
                        slots.append(
                            TextSlot(
                                path=("choices", choice_idx, "message", "content", part_idx, "text"),
                                text=part["text"],
                            )
                        )

        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            slots.append(TextSlot(path=("choices", choice_idx, "delta", "content"), text=delta["content"]))

    return slots


def get_value(payload: dict[str, Any], path: tuple[Any, ...]) -> Any:
    current: Any = payload
    for part in path:
        current = current[part]
    return current


def set_value(payload: dict[str, Any], path: tuple[Any, ...], value: Any) -> None:
    current: Any = payload
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value
