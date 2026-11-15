"""Lifecycle-тесты для привязки SemanticIndexService к app.state.

Итерация 16.3.2 карты №3. Проверяют, что lifespan привязывает
default-сервис к app.state.semantic_index и сохраняет ленивое
построение индекса.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from src.main import lifespan
from src.semantic_index import get_default_semantic_index_service


def _setup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Минимальный env для development-старта без реальных секретов."""
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.delenv("API_SECRET_KEY", raising=False)
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)


@pytest.mark.asyncio
async def test_lifespan_exposes_semantic_index_on_app_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """app.state.semantic_index — тот же экземпляр, что и default-сервис."""
    _setup_env(monkeypatch)
    app_test = FastAPI()
    fake_entries = [{"id": "1", "name": "grammar"}]

    with patch("src.main.PromptBuilder") as mock_builder_cls:
        mock_builder = MagicMock()
        mock_builder.load_full_kb = MagicMock()
        mock_builder_cls.return_value = mock_builder
        with patch("src.main.run_startup_checks"):
            with patch("src.main.load_scoring_weights"):
                with patch(
                    "src.main._collect_semantic_entries",
                    return_value=fake_entries,
                ):
                    async with lifespan(app_test):
                        assert app_test.state.semantic_index is get_default_semantic_index_service()


@pytest.mark.asyncio
async def test_lifespan_does_not_build_index_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Старт не строит индекс: service.get() is None и модель не загружена."""
    _setup_env(monkeypatch)
    app_test = FastAPI()
    fake_entries = [{"id": "1", "name": "grammar"}]

    with patch("src.main.PromptBuilder") as mock_builder_cls:
        mock_builder = MagicMock()
        mock_builder.load_full_kb = MagicMock()
        mock_builder_cls.return_value = mock_builder
        with patch("src.main.run_startup_checks"):
            with patch("src.main.load_scoring_weights"):
                with patch(
                    "src.main._collect_semantic_entries",
                    return_value=fake_entries,
                ):
                    with patch(
                        "src.semantic_index.SemanticIndex._get_model",
                    ) as mock_get_model:
                        async with lifespan(app_test):
                            service = app_test.state.semantic_index
                            assert service.get() is None
                            assert mock_get_model.call_count == 0


@pytest.mark.asyncio
async def test_lifespan_stores_entries_for_lazy_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Записи сохранены, но индекс не построен — ждёт первого ensure()."""
    _setup_env(monkeypatch)
    app_test = FastAPI()
    fake_entries = [{"id": "1", "name": "grammar"}]

    with patch("src.main.PromptBuilder") as mock_builder_cls:
        mock_builder = MagicMock()
        mock_builder.load_full_kb = MagicMock()
        mock_builder_cls.return_value = mock_builder
        with patch("src.main.run_startup_checks"):
            with patch("src.main.load_scoring_weights"):
                with patch(
                    "src.main._collect_semantic_entries",
                    return_value=fake_entries,
                ):
                    async with lifespan(app_test):
                        service = app_test.state.semantic_index
                        assert service.has_entries() is True
                        assert service.get() is None
