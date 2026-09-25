"""
tests/test_semantic_rerank_contract.py
=======================================
Итерация 5 дорожной карты: контрактные тесты для _semantic_rerank
и ленивой загрузки семантической модели.

Проверяют обязательные сценарии:
1. deep search выключен (semantic_weight=0) — entries возвращаются как есть;
2. пустой список кандидатов — модель не грузится;
3. ошибка загрузки модели — retrieval не падает, entries как есть;
4. обычный retrieval без подготовленного индекса — модель не грузится.

Все тесты используют мок SemanticIndex._get_model, без реальной модели
и без сети.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.knowledge_retrieval import _semantic_rerank
from src.semantic_index import reset_semantic_index, set_semantic_entries


@pytest.fixture(autouse=True)
def _clean_semantic_state() -> None:
    """Сбрасываем глобальное состояние semantic index до и после каждого теста."""
    reset_semantic_index()
    yield
    reset_semantic_index()


def test_deep_search_disabled_returns_entries_unchanged() -> None:
    """deep_semantic_search=False (semantic_weight=0) — entries неизменны, модель не грузится."""
    entries = [{"id": "1"}, {"id": "2"}, {"id": "3"}]

    with patch("src.semantic_index.SemanticIndex._get_model") as mock_get_model:
        result = _semantic_rerank(entries, "some query", semantic_weight=0.0)

    assert result == entries
    mock_get_model.assert_not_called()


def test_semantic_rerank_returns_empty_on_empty_candidates() -> None:
    """Пустой список кандидатов — пустой результат, модель не грузится."""
    with patch("src.semantic_index.SemanticIndex._get_model") as mock_get_model:
        result = _semantic_rerank([], "some query", semantic_weight=0.35)

    assert result == []
    mock_get_model.assert_not_called()


def test_semantic_rerank_returns_entries_on_model_load_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ошибка загрузки модели (ImportError из _get_model) — retrieval не падает,
    entries возвращаются как есть.
    """
    monkeypatch.setattr("src.semantic_index._CACHE_PATH", tmp_path / "e.npy")
    monkeypatch.setattr("src.semantic_index._CACHE_META_PATH", tmp_path / "e.json")

    set_semantic_entries([{"id": "1", "name": "a"}])
    entries = [{"id": "1"}, {"id": "2"}]

    with patch(
        "src.semantic_index.SemanticIndex._get_model",
        side_effect=ImportError("sentence-transformers not installed"),
    ):
        result = _semantic_rerank(entries, "some query", semantic_weight=0.35)

    assert result == entries


def test_semantic_rerank_without_entries_does_not_load_model() -> None:
    """
    Обычный retrieval: индекс не подготовлен (нет entries) —
    модель не грузится, entries возвращаются как есть.
    """
    entries = [{"id": "1"}, {"id": "2"}]

    with patch("src.semantic_index.SemanticIndex._get_model") as mock_get_model:
        result = _semantic_rerank(entries, "some query", semantic_weight=0.35)

    assert result == entries
    mock_get_model.assert_not_called()
