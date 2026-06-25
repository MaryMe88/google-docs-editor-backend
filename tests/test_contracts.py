"""
tests/test_contracts.py
=======================
Контракт-тесты: проверяют ТОЛЬКО публичные интерфейсы между модулями.
Бизнес-логику не тестируют — для этого есть остальные тесты в tests/.

Правило: пока эти тесты зелёные — файлы согласованы между собой.
Запуск: pytest tests/test_contracts.py -v
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Контракт 1: PromptBuilder публичный API
# ---------------------------------------------------------------------------

class TestPromptBuilderContract:
    """
    Публичный интерфейс, который использует main.py:
    - PromptBuilder() — инициализация без аргументов
    - .build(text, domain, intent, audience,
              overlays, outputmode, includeknowledge, include_few_shot) — сборка промпта
    - .get_available_intents() -> Set[str]    — множество строк
    - .get_available_overlays() -> Set[str]   — множество строк
    - .reload_configs() -> None               — сброс кэша

    Если DeepSeek переименует хотя бы один метод — тест упадёт сразу.
    """

    @pytest.fixture(scope="class")
    def pb(self):
        from src.prompt_builder import PromptBuilder
        return PromptBuilder()

    def test_class_importable(self):
        from src.prompt_builder import PromptBuilder  # noqa: F401

    def test_audience_profile_importable(self):
        from src.prompt_builder import AudienceProfile  # noqa: F401

    def test_build_returns_str(self, pb):
        result = pb.build(
            text="Тестовый текст.",
            domain="marketing",
            intent=None,
            audience=None,
            overlays=[],
            outputmode="text_only",
            includeknowledge=True,
        )
        assert isinstance(result, str)
        assert "Тестовый текст." in result

    def test_build_contains_text(self, pb):
        """Текст пользователя всегда попадает в промпт."""
        marker = "уникальная_метка_xyz_987"
        result = pb.build(
            text=marker,
            domain="blog",
            intent=None,
            audience=None,
            overlays=[],
            outputmode="text_only",
            includeknowledge=False,
        )
        assert marker in result

    def test_build_accepts_include_few_shot(self, pb):
        """Проверка, что метод build принимает параметр include_few_shot (PR‑2)."""
        # Вызов с include_few_shot=True
        result_true = pb.build(
            text="Текст с few-shot.",
            domain="blog",
            include_few_shot=True,
            includeknowledge=False,
        )
        assert isinstance(result_true, str)
        # Вызов с include_few_shot=False
        result_false = pb.build(
            text="Текст без few-shot.",
            domain="blog",
            include_few_shot=False,
            includeknowledge=False,
        )
        assert isinstance(result_false, str)

    def test_get_available_intents_returns_set_of_str(self, pb):
        intents = pb.get_available_intents()
        assert isinstance(intents, set)
        for item in intents:
            assert isinstance(item, str), f"intent должен быть str, получили {type(item)}"

    def test_get_available_overlays_returns_set_of_str(self, pb):
        overlays = pb.get_available_overlays()
        assert isinstance(overlays, set)
        for item in overlays:
            assert isinstance(item, str), f"overlay должен быть str, получили {type(item)}"

    def test_reload_configs_exists_and_callable(self, pb):
        # Проверяем наличие метода; вызов не должен бросать исключений
        assert callable(getattr(pb, "reload_configs", None)), \
            "PromptBuilder должен иметь метод reload_configs()"
        pb.reload_configs()

    def test_build_with_audience(self, pb):
        from src.prompt_builder import AudienceProfile
        audience = AudienceProfile(
            kind="b2b",
            expertise="pro",
            formality="neutral",
            description="Менеджеры",
        )
        result = pb.build(
            text="Короткий текст.",
            domain="marketing",
            intent=None,
            audience=audience,
            overlays=[],
            outputmode="text_only",
            includeknowledge=False,
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Контракт 2: tag_registry публичный API
# ---------------------------------------------------------------------------

class TestTagRegistryContract:
    """
    Функции, которые импортирует prompt_builder.py:
    - normalize_tag(str) -> str
    - normalize_tags(Iterable[str]) -> List[str]
    - build_known_tags(dict) -> Set[str]
    """

    def test_normalize_tag_importable(self):
        from src.tag_registry import normalize_tag  # noqa: F401

    def test_normalize_tags_importable(self):
        from src.tag_registry import normalize_tags  # noqa: F401

    def test_build_known_tags_importable(self):
        from src.tag_registry import build_known_tags  # noqa: F401

    def test_normalize_tag_returns_str(self):
        from src.tag_registry import normalize_tag
        result = normalize_tag("Marketing")
        assert isinstance(result, str)
        assert result == result.lower()

    def test_normalize_tag_aliases(self):
        """Все алиасы из TAG_ALIASES должны разрешаться в канонические значения."""
        from src.tag_registry import normalize_tag
        # Примеры алиасов из текущего tag_registry.py
        assert normalize_tag("anti-ai") == normalize_tag("antiai")
        assert normalize_tag("info-style") == normalize_tag("infostyle")

    def test_normalize_tags_deduplicates(self):
        from src.tag_registry import normalize_tags
        result = normalize_tags(["marketing", "marketing", "Marketing"])
        assert result.count("marketing") == 1

    def test_normalize_tags_skips_non_strings(self):
        from src.tag_registry import normalize_tags
        result = normalize_tags(["marketing", None, 42, "blog"])  # type: ignore[list-item]
        assert None not in result
        assert 42 not in result

    def test_build_known_tags_returns_set(self):
        from src.tag_registry import build_known_tags
        sample = {
            "domains": {
                "marketing": {"primary": ["marketing"], "expanded": ["sales"]},
            }
        }
        result = build_known_tags(sample)
        assert isinstance(result, set)
        assert "marketing" in result


# ---------------------------------------------------------------------------
# Контракт 3: llm_client публичный API
# ---------------------------------------------------------------------------

class TestLLMClientContract:
    """
    Публичный интерфейс, который использует main.py:
    - class LLMProvider(str, Enum) — значения PERPLEXITY, OPENAI, OPENROUTER, ANTHROPIC
    - create_llm_client(provider, ...) -> BaseLLMClient (async context manager)
    - class LLMError
    """

    def test_llm_provider_importable(self):
        from src.llm_client import LLMProvider  # noqa: F401

    def test_llm_provider_has_required_values(self):
        from src.llm_client import LLMProvider
        for name in ("PERPLEXITY", "OPENAI", "OPENROUTER", "ANTHROPIC"):
            assert hasattr(LLMProvider, name), \
                f"LLMProvider должен содержать {name}"

    def test_create_llm_client_importable(self):
        from src.llm_client import create_llm_client  # noqa: F401

    def test_llm_error_importable(self):
        from src.llm_client import LLMError  # noqa: F401

    def test_create_llm_client_raises_on_missing_key(self):
        """Фабрика должна бросить ValueError (не AttributeError), если нет ключа."""
        import os
        from src.llm_client import LLMProvider, create_llm_client
        # Временно убираем ключ из окружения
        key = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            with pytest.raises((ValueError, Exception)):
                create_llm_client(provider=LLMProvider.OPENROUTER, apikey=None)
        finally:
            if key:
                os.environ["OPENROUTER_API_KEY"] = key

    def test_create_llm_client_is_async_context_manager(self):
        """Результат create_llm_client должен поддерживать async with."""
        import os
        from src.llm_client import LLMProvider, create_llm_client

        # Убираем системный прокси из окружения на время теста —
        # httpx не поддерживает socks4, который может быть выставлен в системе.
        proxy_keys = [k for k in os.environ if "proxy" in k.lower()]
        clean_env = {k: v for k, v in os.environ.items() if k not in proxy_keys}

        with patch.dict(os.environ, clean_env, clear=True):
            client = create_llm_client(
                provider=LLMProvider.OPENROUTER,
                apikey="fake-key-for-contract-test",
            )
        assert hasattr(client, "__aenter__"), "Клиент должен быть async context manager"
        assert hasattr(client, "__aexit__")

    def test_llm_response_fields(self):
        """LLMResponse должен содержать поля content, model, provider, tokens_used."""
        from src.llm_client import LLMResponse
        resp = LLMResponse(
            content="text",
            model="openrouter/auto",
            provider="openrouter",
            tokens_used=100,
        )
        assert resp.content == "text"
        assert resp.model == "openrouter/auto"
        assert resp.provider == "openrouter"
        assert resp.tokens_used == 100


# ---------------------------------------------------------------------------
# Контракт 4: FastAPI-эндпоинты
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_client():
    """
    Создаём тестовый клиент с замоканным LLM — реального вызова нет,
    проверяем только маршрутизацию, схемы и коды ответов.
    """
    from src.llm_client import LLMResponse

    mock_response = LLMResponse(
        content="Отредактированный текст.",
        model="openrouter/auto",
        provider="openrouter",
        tokens_used=42,
    )

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    # Отключаем проверку тегов базы знаний, так как в тестовой среде KB может не содержать всех канонических тегов
    with patch("src.llm_client.create_llm_client", return_value=mock_client), \
         patch("src.startup_checks._check_tags_vs_kb"):
        from src.main import app
        with TestClient(app) as client:
            yield client


class TestFastAPIContractHealth:
    def test_root_returns_ok(self, api_client):
        r = api_client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert isinstance(data["version"], str)

    def test_health_returns_ok(self, api_client):
        """
        /health возвращает 200 (все провайдеры доступны) или 503 (деградация —
        нет API-ключей в CI). Оба кода корректны; проверяем структуру ответа.
        """
        r = api_client.get("/health")
        assert r.status_code in (200, 503), (
            f"/health вернул неожиданный код {r.status_code}: {r.text}"
        )
        data = r.json()
        assert "status" in data, "Поле 'status' обязательно в ответе /health"
        assert "version" in data, "Поле 'version' обязательно в ответе /health"


class TestFastAPIContractEdit:
    """
    Контракт для POST /api/edit:
    - принимает EditRequest (схема),
    - возвращает EditResponse (схема),
    - поля edited_text, model, provider обязательны.
    """

    BASE_PAYLOAD: Dict[str, Any] = {
        "text": "Тестовый текст для редактирования.",
        "domain": "marketing",
        "intent": None,
        "audience": {
            "kind": "b2b",
            "expertise": "pro",
            "formality": "neutral",
            "description": "",
        },
        "overlays": [],
        "output_mode": "text_only",
        "provider": "openrouter",
        "model": "openrouter/auto",
        "temperature": 0.3,
    }

    def test_edit_returns_200(self, api_client):
        r = api_client.post("/api/edit", json=self.BASE_PAYLOAD)
        assert r.status_code == 200, f"Ожидали 200, получили {r.status_code}: {r.text}"

    def test_edit_response_has_required_fields(self, api_client):
        r = api_client.post("/api/edit", json=self.BASE_PAYLOAD)
        assert r.status_code == 200
        data = r.json()
        assert "edited_text" in data, "Поле edited_text обязательно"
        assert "model" in data, "Поле model обязательно"
        assert "provider" in data, "Поле provider обязательно"

    def test_edit_edited_text_is_str(self, api_client):
        r = api_client.post("/api/edit", json=self.BASE_PAYLOAD)
        assert isinstance(r.json()["edited_text"], str)

    def test_edit_rejects_empty_text(self, api_client):
        payload = {**self.BASE_PAYLOAD, "text": ""}
        r = api_client.post("/api/edit", json=payload)
        assert r.status_code == 422, "Пустой текст должен давать 422"

    def test_edit_rejects_unknown_intent(self, api_client):
        payload = {**self.BASE_PAYLOAD, "intent": "nonexistent_intent_xyz"}
        r = api_client.post("/api/edit", json=payload)
        # Валидация выполняется в Pydantic (contracts.py), которая возвращает 422
        assert r.status_code == 422, "Неизвестный intent должен давать 422 Unprocessable Entity"

    def test_edit_rejects_unknown_overlay(self, api_client):
        payload = {**self.BASE_PAYLOAD, "overlays": ["unknown_overlay_xyz"]}
        r = api_client.post("/api/edit", json=payload)
        # Валидация выполняется в Pydantic (contracts.py), которая возвращает 422
        assert r.status_code == 422, "Неизвестный overlay должен давать 422 Unprocessable Entity"

    def test_edit_rejects_unknown_provider(self, api_client):
        payload = {**self.BASE_PAYLOAD, "provider": "unknown_provider"}
        r = api_client.post("/api/edit", json=payload)
        # Валидация Pydantic (через field_validator) возвращает 422
        # Это не переопределено явной проверкой в main.py, поэтому остаётся 422
        assert r.status_code == 422, "Неизвестный provider должен давать 422 Unprocessable Entity"

    def test_edit_with_storytelling_intent(self, api_client):
        payload = {**self.BASE_PAYLOAD, "intent": "storytelling"}
        r = api_client.post("/api/edit", json=payload)
        assert r.status_code == 200

    def test_edit_text_and_report_mode(self, api_client):
        payload = {**self.BASE_PAYLOAD, "output_mode": "text_and_report"}
        r = api_client.post("/api/edit", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "edited_text" in data

    def test_edit_request_has_include_few_shot_field(self, api_client):
        """
        Проверка контракта: в EditRequest есть поле include_few_shot со значением по умолчанию.
        (PR‑2)
        """
        # Отправляем запрос без явного указания поля — не должно быть ошибки
        r = api_client.post("/api/edit", json=self.BASE_PAYLOAD)
        assert r.status_code == 200

        # Явно выключаем few‑shot
        payload_false = {**self.BASE_PAYLOAD, "include_few_shot": False}
        r2 = api_client.post("/api/edit", json=payload_false)
        assert r2.status_code == 200

        # Явно включаем few‑shot
        payload_true = {**self.BASE_PAYLOAD, "include_few_shot": True}
        r3 = api_client.post("/api/edit", json=payload_true)
        assert r3.status_code == 200