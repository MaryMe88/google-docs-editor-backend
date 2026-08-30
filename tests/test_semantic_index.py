# tests/test_semantic_index.py
"""
Тесты для семантического индекса (semantic_index.py) и его интеграции с PromptBuilder.

Проверяют:
- Загрузку KB через PromptBuilder.load_full_kb().
- Использование _loaded_kb в _collect_semantic_entries.
- Вызов init_semantic_index с правильными записями.
- Поведение при пустой KB.
- Сброс индекса при reload_configs.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI

from src.prompt_builder import PromptBuilder
from src.semantic_index import SemanticIndex, init_semantic_index, get_semantic_index


# ============================================================================
# Фикстуры
# ============================================================================

@pytest.fixture
def mock_semantic_index() -> MagicMock:
    """Создаёт мок для SemanticIndex."""
    mock = MagicMock(spec=SemanticIndex)
    mock.build = MagicMock()
    mock.search = MagicMock(return_value=[])
    mock.is_ready = MagicMock(return_value=True)
    return mock


# ============================================================================
# Тесты для PromptBuilder.load_full_kb()
# ============================================================================

def test_load_full_kb_loads_kb(builder: PromptBuilder) -> None:
    """Проверяет, что load_full_kb загружает KB и сохраняет в _loaded_kb."""
    assert builder._loaded_kb is None
    kb = builder.load_full_kb()
    assert kb is not None
    # Проверяем, что _loaded_kb установлен
    assert builder._loaded_kb is kb
    # Проверяем, что KB содержит ожидаемые блоки (хотя бы grammar_errors)
    assert hasattr(kb, "grammar_errors"), "KB должна содержать атрибут grammar_errors"
    # Повторный вызов возвращает тот же объект
    assert builder.load_full_kb() is kb


def test_reload_configs_clears_loaded_kb(builder: PromptBuilder) -> None:
    """Проверяет, что reload_configs сбрасывает _loaded_kb."""
    builder.load_full_kb()
    assert builder._loaded_kb is not None
    builder.reload_configs()
    assert builder._loaded_kb is None


# ============================================================================
# Тесты для _collect_semantic_entries
# ============================================================================

def test_collect_semantic_entries_uses_loaded_kb() -> None:
    """Проверяет, что _collect_semantic_entries использует _loaded_kb, а не устаревший 'kb'."""
    from src.main import _collect_semantic_entries

    # Создаём мок для app.state с PromptBuilder
    # ИСПРАВЛЕНИЕ: убрали spec=PromptBuilder, чтобы атрибут _loaded_kb не блокировался
    mock_pb = MagicMock()
    # Имитируем загруженную KB с записями
    mock_kb = MagicMock()
    mock_kb.grammar_errors = [{"wrong": "test", "id": "1"}]
    mock_kb.stylistic_issues = []
    mock_kb.logic_issues = []
    mock_pb._loaded_kb = mock_kb

    app = MagicMock()
    app.state = MagicMock()
    app.state.prompt_builder = mock_pb

    entries = _collect_semantic_entries(app)
    # Должна быть одна запись из grammar_errors
    assert len(entries) == 1
    assert entries[0]["wrong"] == "test"


def test_collect_semantic_entries_returns_empty_if_no_loaded_kb() -> None:
    """Если _loaded_kb отсутствует, возвращается пустой список и логируется предупреждение."""
    from src.main import _collect_semantic_entries

    mock_pb = MagicMock(spec=PromptBuilder)
    # Нет _loaded_kb
    mock_pb._loaded_kb = None

    app = MagicMock()
    app.state = MagicMock()
    app.state.prompt_builder = mock_pb

    with patch("src.main.logger.warning") as mock_warning:
        entries = _collect_semantic_entries(app)
        assert entries == []
        mock_warning.assert_called_once_with("SemanticIndex: KB не загружена в PromptBuilder")


# ============================================================================
# Тесты для _build_semantic_index_background
# ============================================================================

@pytest.mark.asyncio
async def test_build_semantic_index_background_calls_init_with_entries() -> None:
    """Проверяет, что фоновая задача вызывает init_semantic_index с собранными записями."""
    from src.main import _build_semantic_index_background

    # Создаём мок для _collect_semantic_entries, который возвращает список записей
    mock_entries = [{"wrong": "test", "id": "1"}]
    with patch("src.main._collect_semantic_entries", return_value=mock_entries) as mock_collect:
        # ИСПРАВЛЕНИЕ: заменяем AsyncMock на MagicMock, т.к. init_semantic_index синхронная
        with patch("src.main.init_semantic_index") as mock_init:
            app = MagicMock()
            app.state.semantic_index_status = "not_started"

            await _build_semantic_index_background(app)

            mock_collect.assert_called_once_with(app)
            mock_init.assert_called_once_with(mock_entries)
            assert app.state.semantic_index_status == "ready"


@pytest.mark.asyncio
async def test_build_semantic_index_background_skips_if_no_entries() -> None:
    """Если _collect_semantic_entries возвращает пустой список, индекс не строится."""
    from src.main import _build_semantic_index_background

    with patch("src.main._collect_semantic_entries", return_value=[]) as mock_collect:
        with patch("src.main.init_semantic_index") as mock_init:
            with patch("src.main.logger.warning") as mock_warning:
                app = MagicMock()
                app.state.semantic_index_status = "not_started"

                await _build_semantic_index_background(app)

                mock_collect.assert_called_once_with(app)
                mock_init.assert_not_called()
                mock_warning.assert_called_once_with("SemanticIndex: нет записей для индексации, индекс не строится")
                assert app.state.semantic_index_status == "ready"


@pytest.mark.asyncio
async def test_build_semantic_index_background_handles_exception() -> None:
    """Проверяет, что исключение в init_semantic_index логируется и статус становится 'failed'."""
    from src.main import _build_semantic_index_background

    mock_entries = [{"wrong": "test"}]
    with patch("src.main._collect_semantic_entries", return_value=mock_entries):
        with patch("src.main.init_semantic_index", side_effect=Exception("Test error")):
            with patch("src.main.logger.error") as mock_error:
                app = MagicMock()
                app.state.semantic_index_status = "not_started"

                await _build_semantic_index_background(app)

                assert app.state.semantic_index_status == "failed"
                assert app.state.semantic_index_error == "Test error"
                mock_error.assert_called_once()


# ============================================================================
# Тесты для lifespan интеграции
# ============================================================================

@pytest.mark.asyncio
async def test_lifespan_calls_load_full_kb() -> None:
    """Проверяет, что в lifespan вызывается load_full_kb() перед построением индекса."""
    from src.main import lifespan

    # Создаём мок для PromptBuilder
    mock_builder = MagicMock(spec=PromptBuilder)
    mock_builder.load_full_kb = MagicMock(return_value=MagicMock())
    mock_builder.startup_check = MagicMock()
    mock_builder.get_available_intents = MagicMock(return_value=set())
    mock_builder.get_available_overlays = MagicMock(return_value=set())

    with patch("src.main.PromptBuilder", return_value=mock_builder):
        with patch("src.main.run_startup_checks"):
            with patch("src.main.load_scoring_weights"):
                with patch("src.main._build_semantic_index_background", new_callable=AsyncMock):
                    # ИСПРАВЛЕНИЕ: добавили API_SECRET_KEY и DISABLE_SEMANTIC_INDEX=false,
                    # чтобы гарантировать, что производственный путь создания задачи проверяется.
                    with patch.dict(
                        os.environ,
                        {
                            "OPENROUTER_API_KEY": "test-key",
                            "API_SECRET_KEY": "test-secret",
                            "DISABLE_SEMANTIC_INDEX": "false",
                        },
                    ):
                        app = FastAPI()
                        async with lifespan(app):
                            pass

    mock_builder.load_full_kb.assert_called_once()


# ============================================================================
# Тесты для самого SemanticIndex (лёгкие, без реальной модели)
# ============================================================================

# ИСПРАВЛЕНИЕ: добавлены параметры tmp_path и monkeypatch, изолирован кэш
def test_semantic_index_build_caches_embeddings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверяет, что SemanticIndex строит и кеширует эмбеддинги (без реальной модели)."""
    # Изолируем кэш-путь, чтобы не использовать существующий кэш
    monkeypatch.setattr("src.semantic_index._CACHE_PATH", tmp_path / "embeddings.npy")
    monkeypatch.setattr("src.semantic_index._CACHE_META_PATH", tmp_path / "embeddings_meta.json")

    # Мокаем _get_model, чтобы не загружать реальную модель
    with patch("src.semantic_index.SemanticIndex._get_model") as mock_get_model:
        mock_model = MagicMock()
        # Возвращаем list, но build должен преобразовать в ndarray
        mock_model.encode = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        mock_get_model.return_value = mock_model

        index = SemanticIndex()
        entries = [{"name": "test1", "description": "desc1"}, {"name": "test2", "description": "desc2"}]
        index.build(entries)

        # Проверяем, что encode вызван с правильными текстами
        mock_model.encode.assert_called_once()
        # Проверяем, что эмбеддинги сохранены как numpy.ndarray
        assert index.embeddings is not None
        assert isinstance(index.embeddings, np.ndarray)
        assert index.embeddings.shape == (2, 2)
        assert index.embeddings.dtype == np.float32


def test_semantic_index_search_returns_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Поиск возвращает запись с наибольшим similarity score."""
    monkeypatch.setattr(
        "src.semantic_index._CACHE_PATH",
        tmp_path / "embeddings.npy",
    )
    monkeypatch.setattr(
        "src.semantic_index._CACHE_META_PATH",
        tmp_path / "embeddings_meta.json",
    )

    with patch(
        "src.semantic_index.SemanticIndex._get_model"
    ) as mock_get_model:
        mock_model = MagicMock()

        record_embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        query_embedding = np.asarray(
            [
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )

        mock_model.encode = MagicMock(
            side_effect=[
                record_embeddings,
                query_embedding,
            ]
        )
        mock_get_model.return_value = mock_model

        index = SemanticIndex()
        entries = [
            {"id": "1", "name": "one"},
            {"id": "2", "name": "two"},
        ]

        index.build(entries, force_rebuild=True)
        results = index.search("test query", top_k=1)

    assert len(results) == 1
    entry, score = results[0]

    assert entry["id"] == "1"
    assert score == pytest.approx(1.0)


def test_semantic_index_is_ready() -> None:
    """Проверяет is_ready."""
    index = SemanticIndex()
    assert index.is_ready() is False
    index._is_built = True
    index.embeddings = np.array([[1.0]])
    assert index.is_ready() is True


# ============================================================================
# Тесты для глобального индекса (get_semantic_index, init_semantic_index)
# ============================================================================

def test_get_semantic_index_returns_none_if_not_initialized() -> None:
    """Проверяет, что get_semantic_index возвращает None, если индекс не инициализирован."""
    from src.semantic_index import _global_index, get_semantic_index
    # Сохраняем старый, чтобы не сломать другие тесты
    old_global = _global_index
    try:
        # Устанавливаем в None
        import src.semantic_index as si
        si._global_index = None
        assert get_semantic_index() is None
    finally:
        si._global_index = old_global


def test_init_semantic_index_sets_global_index() -> None:
    """Проверяет, что init_semantic_index создаёт и сохраняет глобальный индекс."""
    from src.semantic_index import init_semantic_index, get_semantic_index

    with patch("src.semantic_index.SemanticIndex.build") as mock_build:
        # Мокаем _get_model для ускорения
        with patch("src.semantic_index.SemanticIndex._get_model") as mock_get_model:
            mock_get_model.return_value = MagicMock()

            entries = [{"id": "1"}]
            index = init_semantic_index(entries, force_rebuild=True)
            assert index is not None
            # Проверяем, что глобальный индекс установлен через get_semantic_index()
            assert get_semantic_index() is index
            mock_build.assert_called_once_with(entries, force_rebuild=True)