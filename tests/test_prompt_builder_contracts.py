"""Тесты контрактов PromptBuilder (итерация 15.1 карты №3)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from src.config_types import AudienceProfile, KnowledgeLevel
from src.prompt_builder import BuildOptions, BuildRequest, BuildResult


def test_build_options_defaults_match_build_signature() -> None:
    """Дефолты BuildOptions совпадают с дефолтами PromptBuilder.build()."""
    opts = BuildOptions()
    assert opts.include_knowledge is True
    assert opts.include_few_shot is True
    assert opts.knowledge_level is KnowledgeLevel.STANDARD
    assert opts.token_budget is None
    assert opts.include_retrieval_meta is False
    assert opts.few_shot_seed is None
    assert opts.deep_semantic_search is False


def test_build_request_minimal_uses_defaults() -> None:
    """Минимальный BuildRequest: только text/domain, остальное из defaults."""
    req = BuildRequest(text="t", domain="d")
    assert req.text == "t"
    assert req.domain == "d"
    assert req.intent is None
    assert req.audience is None
    assert req.overlays is None
    assert req.output_mode == "text_only"
    assert req.options == BuildOptions()


def test_build_result_prompt_and_meta() -> None:
    """BuildResult: prompt обязателен, retrieval_meta опционален."""
    r1 = BuildResult(prompt="hello")
    assert r1.prompt == "hello"
    assert r1.retrieval_meta is None

    r2 = BuildResult(prompt="hello", retrieval_meta={"a": 1})
    assert r2.prompt == "hello"
    assert r2.retrieval_meta == {"a": 1}


def test_contracts_reuse_existing_types() -> None:
    """BuildRequest использует существующие AudienceProfile и KnowledgeLevel."""
    audience = AudienceProfile(kind="b2b", expertise="pro", formality="neutral")
    req = BuildRequest(
        text="t",
        domain="d",
        audience=audience,
        options=BuildOptions(knowledge_level=KnowledgeLevel.FULL),
    )
    assert req.audience is audience
    assert req.options.knowledge_level is KnowledgeLevel.FULL


def test_contracts_are_frozen() -> None:
    """Все три контракта — frozen dataclass."""
    with pytest.raises(FrozenInstanceError):
        BuildOptions().include_knowledge = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        BuildRequest(text="t", domain="d").text = "x"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        BuildResult(prompt="p").prompt = "q"  # type: ignore[misc]


def test_build_request_has_no_legacy_kwargs_field() -> None:
    """BuildRequest — чистый контракт: legacy_kwargs не входит в поля."""
    field_names = {f.name for f in fields(BuildRequest)}
    assert "legacy_kwargs" not in field_names
    assert field_names == {
        "text",
        "domain",
        "intent",
        "audience",
        "overlays",
        "output_mode",
        "options",
    }
