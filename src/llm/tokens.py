"""Оценка max_tokens по длине промпта.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

from ._constants import (
    _CHARS_PER_TOKEN,
    _MAX_MAX_TOKENS,
    _MIN_MAX_TOKENS,
    _RESPONSE_BUDGET_MULTIPLIER,
)


def estimate_max_tokens(prompt: str) -> int:
    """Оценивает max_tokens ответа по длине промпта: пропорционально длине, с ограничением снизу и сверху."""
    prompt_length = max(len(prompt), 0)
    estimated_input_tokens = prompt_length // _CHARS_PER_TOKEN
    response_budget = int(estimated_input_tokens * _RESPONSE_BUDGET_MULTIPLIER)

    if response_budget < _MIN_MAX_TOKENS:
        return _MIN_MAX_TOKENS
    if response_budget > _MAX_MAX_TOKENS:
        return _MAX_MAX_TOKENS
    return response_budget
