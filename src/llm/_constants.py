"""Константы для оценки max_tokens и бюджета ответа.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

_DEFAULT_MAX_TOKENS = 6000
_MIN_MAX_TOKENS = 1536
_MAX_MAX_TOKENS = 12000
_CHARS_PER_TOKEN = 4
_RESPONSE_BUDGET_MULTIPLIER = 1.35
