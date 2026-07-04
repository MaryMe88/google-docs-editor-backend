"""Интеграционные тесты защитного слоя от плейсхолдеров на /api/edit.

Сценарии (см. ТЗ, «Что нужно проверить после исправления»):
1. Текст без PII проходит без токенов в финальном ответе.
2. Ответ с плейсхолдером на первой попытке восстанавливается retry
   (вторая попытка возвращает чистый текст) — пользователь получает 200
   и текст без токенов.
3. Fail-closed: если токены остаются и после retry, возвращается 502,
   а не текст со служебными маркерами.
4. Ложноположительный кейс: легальные скобки/русский текст не триггерят guard.
"""

from __future__ import annotations

import os
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

_TEST_API_KEY = "test-secret-key-guard"

BASE_PAYLOAD: Dict[str, Any] = {
    "text": "Тестовый текст для редактирования.",
    "domain": "marketing",
    "intent": None,
    "audience": {"kind": "b2b", "expertise": "pro", "formality": "neutral", "description": ""},
    "overlays": [],
    "output_mode": "text_only",
    "provider": "openrouter",
    "model": "openrouter/auto",
    "temperature": 0.3,
}


def _make_response(content: str):
    """Создаёт объект LLMResponse с заданным content."""
    from src.llm_client import LLMResponse

    return LLMResponse(
        content=content,
        model="openrouter/auto",
        provider="openrouter",
        tokens_used=10,
    )


def _client_with_llm(side_effect):
    """Собирает TestClient с замоканным call_with_fallback.

    Args:
        side_effect: список/функция для AsyncMock, задающий ответы LLM
            по порядку вызовов.
    """
    mock_call = AsyncMock(side_effect=side_effect)
    # Мокаем именно ссылку, импортированную в src.main.
    patches = [
        patch("src.main.call_with_fallback", mock_call),
        patch("src.startup_checks._check_tags_vs_kb"),
        patch.dict(os.environ, {"API_SECRET_KEY": _TEST_API_KEY}),
    ]
    return mock_call, patches


def _run(side_effect, payload: Dict[str, Any]):
    mock_call, patches = _client_with_llm(side_effect)
    for p in patches:
        p.start()
    try:
        from src.main import app

        with TestClient(app) as client:
            resp = client.post(
                "/api/edit", json=payload, headers={"X-API-Key": _TEST_API_KEY}
            )
        return resp, mock_call
    finally:
        for p in reversed(patches):
            p.stop()


# ---------------------------------------------------------------------------
# 1. Чистый текст — один вызов, 200, без токенов
# ---------------------------------------------------------------------------
def test_clean_output_single_call() -> None:
    resp, mock_call = _run(
        side_effect=[_make_response("Чистый отредактированный текст.")],
        payload=BASE_PAYLOAD,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["edited_text"] == "Чистый отредактированный текст."
    assert mock_call.await_count == 1  # retry не потребовался


# ---------------------------------------------------------------------------
# 2. Плейсхолдер на первой попытке -> retry -> чистый результат
# ---------------------------------------------------------------------------
def test_placeholder_recovered_by_retry() -> None:
    resp, mock_call = _run(
        side_effect=[
            _make_response("[PERSON_NAME] важно учесть заранее."),
            _make_response("Марии важно учесть это заранее."),
        ],
        payload=BASE_PAYLOAD,
    )
    assert resp.status_code == 200, resp.text
    edited = resp.json()["edited_text"]
    assert edited == "Марии важно учесть это заранее."
    assert "[PERSON_NAME]" not in edited
    assert mock_call.await_count == 2  # был ровно один retry


# ---------------------------------------------------------------------------
# 3. Fail-closed: токены и после retry -> 502, без утечки токенов в ответ
# ---------------------------------------------------------------------------
def test_placeholder_fail_closed() -> None:
    resp, mock_call = _run(
        side_effect=[
            _make_response("[PERSON_NAME] важно учесть заранее."),
            _make_response("[PERSON_NAME] всё ещё здесь."),
        ],
        payload=BASE_PAYLOAD,
    )
    assert resp.status_code == 502, resp.text
    assert "[PERSON_NAME]" not in resp.text  # токен не утёк пользователю
    assert mock_call.await_count == 2


# ---------------------------------------------------------------------------
# 4. Ложноположительный кейс: легальные скобки не триггерят guard
# ---------------------------------------------------------------------------
def test_false_positive_not_triggered() -> None:
    resp, mock_call = _run(
        side_effect=[_make_response("Примечание [см. выше] и русская метка [ВАЖНО].")],
        payload=BASE_PAYLOAD,
    )
    assert resp.status_code == 200, resp.text
    assert mock_call.await_count == 1  # ни одного retry
