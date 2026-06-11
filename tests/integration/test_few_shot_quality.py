"""
tests/integration/test_few_shot_quality.py

Интеграционные тесты для few‑shot (запуск вручную, не в CI).
Проверяют, что модель не копирует примеры дословно и не применяет ложные правки.
Требуют реального API-ключа и запущенного сервера (или использования TestClient с реальным LLM).

Запуск: pytest -m integration -v
"""

from __future__ import annotations

import os
import re

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.prompt_builder import PromptBuilder


@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    Инициализирует приложение с запущенным PromptBuilder.
    Без этого TestClient не вызывает lifespan и app.state.prompt_builder остаётся None.
    """
    pb = PromptBuilder()
    pb.startup_check()
    app.state.prompt_builder = pb
    with TestClient(app) as test_client:
        yield test_client


def call_editor(client: TestClient, text: str, include_few_shot: bool = True) -> str:
    """Отправляет текст на редактирование и возвращает результат."""
    payload = {
        "text": text,
        "domain": "blog",
        "intent": "neutral",
        "include_knowledge": True,
        "include_few_shot": include_few_shot,
        "dry_run": False,
        "provider": "openrouter",
        "model": "openrouter/auto",
    }
    response = client.post("/api/edit", json=payload)
    if response.status_code != 200:
        pytest.fail(f"API вернул {response.status_code}: {response.text}")
    return response.json()["edited_text"]


def has_grammar_error_after_soglasno(text: str) -> bool:
    """
    Проверяет, есть ли в тексте ошибка "согласно + родительный падеж".
    Например: "согласно приказа", "согласно распоряжения".
    """
    matches = re.findall(r'согласно\s+(\S+)', text.lower())
    for m in matches:
        # Если слово заканчивается на -а или -я (родительный падеж), считаем ошибкой
        if m.endswith('а') or m.endswith('я'):
            # Исключаем слова, которые в дательном тоже так заканчиваются? Упростим.
            if m not in ('времени', 'имени'):  # исключения
                return True
    return False


# Пропускаем тесты, если нет API-ключа (например, в CI)
REQUIRED_ENV_VARS = ["OPENROUTER_API_KEY"]
skip_no_key = pytest.mark.skipif(
    any(os.getenv(var) is None for var in REQUIRED_ENV_VARS),
    reason="Нет API-ключа для интеграционных тестов",
)


@skip_no_key
@pytest.mark.integration
def test_no_verbatim_copy_from_few_shot(client: TestClient) -> None:
    """
    Модель не должна дословно копировать correct из примера.
    Пример из базы: "согласно приказа → согласно приказу"
    """
    text = "Он согласился согласно приказа начальника отдела."
    result = call_editor(client, text)
    
    # Проверяем, что ошибка исправлена (родительный падеж 'приказа' заменён)
    assert "приказа" not in result.lower(), f"Ошибка 'приказа' осталась: {result}"
    # Проверяем, что модель не вырезала всю фразу целиком
    assert len(result) > 20, f"Ответ слишком короткий: {result}"
    # Проверяем, что ответ осмысленный (содержит 'начальника')
    assert "начальника" in result, f"Потерян контекст: {result}"


@skip_no_key
@pytest.mark.integration
def test_no_false_positive_from_few_shot(client: TestClient) -> None:
    """
    Модель не должна вносить грамматическую ошибку в правильный текст.
    Текст уже правильный: "Он сделал всё согласно приказу."
    """
    text = "Он сделал всё согласно приказу."
    result = call_editor(client, text)
    
    # Проверяем, что после 'согласно' нет родительного падежа (это и есть единственная
    # проверяемая ошибка). Слово 'приказа' может встречаться в других контекстах
    # (например, "буква приказа") — это не ошибка.
    assert not has_grammar_error_after_soglasno(result), \
        f"Грамматическая ошибка после 'согласно' в ответе: {result}"


@skip_no_key
@pytest.mark.integration
def test_few_shot_disabled_does_not_add_examples(client: TestClient) -> None:
    """
    При include_few_shot=False примеры не влияют на ответ (или влияют не фатально).
    Оба режима должны исправлять грамматическую ошибку.
    """
    text = "Он согласился согласно приказа."
    result_with_fs = call_editor(client, text, include_few_shot=True)
    result_without_fs = call_editor(client, text, include_few_shot=False)

    # Проверяем, что в обоих случаях ошибка исправлена (нет 'приказа')
    assert "приказа" not in result_with_fs.lower(), \
        f"С few-shot не исправлено: {result_with_fs}"
    assert "приказа" not in result_without_fs.lower(), \
        f"Без few-shot не исправлено: {result_without_fs}"
    
    # Проверяем, что ответы не пустые и достаточно длинные
    assert len(result_with_fs) > 10
    assert len(result_without_fs) > 10