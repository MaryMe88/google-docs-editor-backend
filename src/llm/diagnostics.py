"""Диагностика ответов LLM без утечки пользовательских данных.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

import random
from typing import Any


def _safe_str(value: Any, max_len: int = 64) -> str | None:
    """Безопасно приводит примитив к строке, обрезая по длине.

    Возвращает None для объектов сложной структуры (dict, list, tuple),
    чтобы случайно не сериализовать целые ответы или блоки контента.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, bool):
        return None
    elif isinstance(value, int | float):
        text = str(value)
    else:
        return None
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _safe_int(value: Any) -> int | None:
    """Безопасно извлекает int (bool игнорируется — он подкласс int)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _collect_response_diagnostics(
    data: Any,
    *,
    reason_code: str,
    requested_model: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Собирает технические метаданные ответа без пользовательских данных.

    Логируются только: id, фактическая модель, провайдер, количество choices,
    наличие message, finish_reason, тип и длина content, наличие и длина
    reasoning, top-level error (только code/type), счётчики токенов.
    Содержимое ответа, промпт и текст пользователя не покидают эту функцию.
    """
    diag: dict[str, Any] = {
        "reason_code": reason_code,
        "requested_model": requested_model,
        "max_tokens": max_tokens,
    }

    if not isinstance(data, dict):
        diag["data_type"] = type(data).__name__
        return diag

    diag["response_id"] = _safe_str(data.get("id"))
    diag["actual_model"] = _safe_str(data.get("model"))
    diag["provider"] = _safe_str(data.get("provider"))

    has_error = "error" in data
    diag["has_top_level_error"] = has_error
    err = data.get("error")
    if isinstance(err, dict):
        diag["error_code"] = _safe_str(err.get("code"))
        diag["error_type"] = _safe_str(err.get("type"))
    else:
        diag["error_code"] = None
        diag["error_type"] = None

    choices = data.get("choices")
    if isinstance(choices, list):
        diag["choices_count"] = len(choices)
    else:
        diag["choices_count"] = None

    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            diag["finish_reason"] = _safe_str(first.get("finish_reason"))
            message = first.get("message")
            diag["has_message"] = isinstance(message, dict)
            if isinstance(message, dict):
                content = message.get("content")
                diag["content_type"] = type(content).__name__
                diag["content_length"] = len(content) if isinstance(content, str) else None
                reasoning = message.get("reasoning")
                if reasoning is None:
                    diag["has_reasoning"] = False
                    diag["reasoning_length"] = None
                else:
                    diag["has_reasoning"] = True
                    diag["reasoning_length"] = (
                        len(reasoning) if isinstance(reasoning, str) else None
                    )

    usage = data.get("usage")
    if isinstance(usage, dict):
        diag["prompt_tokens"] = _safe_int(usage.get("prompt_tokens"))
        diag["completion_tokens"] = _safe_int(usage.get("completion_tokens"))
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            diag["reasoning_tokens"] = _safe_int(completion_details.get("reasoning_tokens"))
        else:
            diag["reasoning_tokens"] = None

    return diag


def _format_diagnostics(
    diagnostics: dict[str, Any] | None,
    *,
    attempt: int,
) -> str:
    """Формирует компактную строку диагностики для логирования."""
    parts = [f"attempt={attempt}"]
    if diagnostics:
        for key, value in diagnostics.items():
            parts.append(f"{key}={value}")
    return "[" + ", ".join(parts) + "]"


def _backoff_with_jitter(base_delay: float, attempt: int) -> float:
    cap = base_delay * (2**attempt)
    return random.uniform(0, cap)
