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
# Тесты для _collect_semantic_entries (перенесены из test_main, чтобы не дублировать)
# ============================================================================

def test_collect_semantic_entries_uses_loaded_kb() -> None:
    """Проверяет, что _collect_semantic_entries использует _loaded_kb, а не устаревший 'kb'."""
    from src.main import _collect_semantic_entries

    mock_pb = MagicMock()
    mock_kb = MagicMock()
    mock_kb.grammar_errors = [{"wrong": "test", "id": "1"}]
    mock_kb.stylistic_issues = []
    mock_kb.logic_issues = []
    mock_pb._loaded_kb = mock_kb

    app = MagicMock()
    app.state = MagicMock()
    app.state.prompt_builder = mock_pb

    entries = _collect_semantic_entries(app)
    assert len(entries) == 1
    assert entries[0]["wrong"] == "test"


def test_collect_semantic_entries_returns_empty_if_no_loaded_kb() -> None:
    """Если _loaded_kb отсутствует, возвращается пустой список и логируется предупреждение."""
    from src.main import _collect_semantic_entries

    mock_pb = MagicMock(spec=PromptBuilder)
    mock_pb._loaded_kb = None

    app = MagicMock()
    app.state = MagicMock()
    app.state.prompt_builder = mock_pb

    with patch("src.main.logger.warning") as mock_warning:
        entries = _collect_semantic_entries(app)
        assert entries == []
        mock_warning.assert_called_once_with("SemanticIndex: KB не загружена в PromptBuilder")


# ============================================================================
# Тесты для самого SemanticIndex (лёгкие, без реальной модели)
# ============================================================================

def test_semantic_index_build_caches_embeddings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверяет, что SemanticIndex строит и кеширует эмбеддинги (без реальной модели)."""
    monkeypatch.setattr("src.semantic_index._CACHE_PATH", tmp_path / "embeddings.npy")
    monkeypatch.setattr("src.semantic_index._CACHE_META_PATH", tmp_path / "embeddings_meta.json")

    with patch("src.semantic_index.SemanticIndex._get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        mock_get_model.return_value = mock_model

        index = SemanticIndex()
        entries = [{"name": "test1", "description": "desc1"}, {"name": "test2", "description": "desc2"}]
        index.build(entries)

        mock_model.encode.assert_called_once()
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
    old_global = _global_index
    try:
        import src.semantic_index as si
        si._global_index = None
        assert get_semantic_index() is None
    finally:
        si._global_index = old_global


def test_init_semantic_index_sets_global_index() -> None:
    """Проверяет, что init_semantic_index создаёт и сохраняет глобальный индекс."""
    from src.semantic_index import init_semantic_index, get_semantic_index

    with patch("src.semantic_index.SemanticIndex.build") as mock_build:
        with patch("src.semantic_index.SemanticIndex._get_model") as mock_get_model:
            mock_get_model.return_value = MagicMock()

            entries = [{"id": "1"}]
            index = init_semantic_index(entries, force_rebuild=True)
            assert index is not None
            assert get_semantic_index() is index
            mock_build.assert_called_once_with(entries, force_rebuild=True)


# ============================================================================
# Тесты для set_semantic_entries (новая функция)
# ============================================================================

def test_set_semantic_entries_stores_entries() -> None:
    """Проверяет, что set_semantic_entries сохраняет записи в глобальную переменную."""
    from src.semantic_index import set_semantic_entries, _entries_for_index

    # Сохраняем старое значение, чтобы восстановить после теста
    old = _entries_for_index
    try:
        # Устанавливаем явно None, чтобы избежать влияния других тестов
        import src.semantic_index as si
        si._entries_for_index = None

        test_entries = [{"id": "test"}]
        set_semantic_entries(test_entries)
        assert si._entries_for_index is test_entries
    finally:
        si._entries_for_index = old  # восстанавливаем