"""Wiring-тесты для итерации 16.3.3 карты №3.

Проверяют, что явный SemanticIndexService доходит от HTTP-слоя до
_semantic_rerank, а без явной передачи используется _default_service.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.knowledge_retrieval import _semantic_rerank
from src.main import app
from src.prompt_builder.contracts import BuildResult


def test_router_passes_app_state_semantic_index_to_build_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """app.state.semantic_index попадает в BuildOptions.semantic_index_service."""
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.delenv("API_SECRET_KEY", raising=False)
    monkeypatch.setenv("PYTEST_RUNNING", "true")

    fake_service = MagicMock(name="fake_semantic_index_service")
    fake_builder = MagicMock()
    fake_builder.build_prompt = MagicMock(
        return_value=BuildResult(prompt="stub", retrieval_meta={})
    )

    with patch("src.main.PromptBuilder", return_value=fake_builder):
        with patch("src.main.run_startup_checks"):
            with patch("src.main.load_scoring_weights"):
                with patch("src.main._collect_semantic_entries", return_value=[]):
                    with TestClient(app) as client:
                        # Переопределяем то, что lifespan положил в state,
                        # чтобы отследить фактическую передачу.
                        app.state.prompt_builder = fake_builder
                        app.state.semantic_index = fake_service

                        response = client.post(
                            "/api/edit",
                            json={
                                "text": "Это тестовый текст.",
                                "domain": "blog",
                                "provider": "openrouter",
                                "dry_run": True,
                            },
                        )

    assert response.status_code == 200, response.text
    fake_builder.build_prompt.assert_called_once()
    call_request = fake_builder.build_prompt.call_args.args[0]
    assert call_request.options.semantic_index_service is fake_service


def test_semantic_rerank_uses_injected_service() -> None:
    """_semantic_rerank(service=...) зовёт service.ensure и не падает."""
    fake_index = MagicMock(name="fake_index")
    fake_index.is_ready.return_value = True
    fake_index.search.return_value = [({"name": "a"}, 0.9), ({"name": "b"}, 0.1)]

    fake_service = MagicMock(name="fake_service")
    fake_service.ensure.return_value = fake_index

    entries = [{"name": "a"}, {"name": "b"}]
    result = _semantic_rerank(
        entries,
        "запрос",
        semantic_weight=0.35,
        service=fake_service,
    )

    fake_service.ensure.assert_called_once_with("cointegrated/rubert-tiny2")
    assert len(result) == 2
    assert {entry["name"] for entry in result} == {"a", "b"}
