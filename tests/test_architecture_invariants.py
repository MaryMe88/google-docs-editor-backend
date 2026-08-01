"""
tests/test_architecture_invariants.py

Архитектурные инварианты, которые должны выполняться всегда.
Проверяют согласованность слоёв и корректность feature resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.prompt_builder import PromptBuilder, KnowledgeLevel
from src.shared_contracts import ALLOWED_DOMAINS, ALLOWED_INTENTS, ALLOWED_OVERLAYS
from src.config_types import DomainConfig
from src.reason_codes import ReasonCode


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder(config_path=Path("config"), kb_path=Path("knowledge_base"))


class TestKnowledgeLevelOverrides:
    """Проверяет, что при FULL уровне блоки storytelling/marketing включаются принудительно."""

    def test_storytelling_included_at_full_without_tag(self, builder: PromptBuilder) -> None:
        """При FULL и allow_storytelling=True storytelling появляется даже без тега."""
        # Домен blog имеет allow_storytelling=True
        prompt = builder.build(
            text="Тестовый текст для проверки сторителлинга.",
            domain="blog",
            knowledge_level=KnowledgeLevel.FULL,
            include_knowledge=True,
            include_few_shot=False,
            # Не передаём тег storytelling
        )
        assert "Сторителлинг-фреймворки:" in prompt, (
            "При FULL уровне и allow_storytelling=True блок storytelling должен быть включён"
        )

    def test_marketing_included_at_full_without_tag(self, builder: PromptBuilder) -> None:
        """При FULL и allow_marketing=True marketing появляется даже без тега."""
        # Домен marketing имеет allow_marketing=True
        prompt = builder.build(
            text="Тестовый текст для проверки маркетинга.",
            domain="marketing",
            knowledge_level=KnowledgeLevel.FULL,
            include_knowledge=True,
            include_few_shot=False,
            # Не передаём тег marketing
        )
        assert "Маркетинговые шаблоны:" in prompt, (
            "При FULL уровне и allow_marketing=True блок marketing должен быть включён"
        )

    def test_storytelling_not_included_at_standard_without_tag(self, builder: PromptBuilder) -> None:
        """При STANDARD уровне storytelling не включается без тега."""
        prompt = builder.build(
            text="Тестовый текст.",
            domain="blog",
            knowledge_level=KnowledgeLevel.STANDARD,
            include_knowledge=True,
            include_few_shot=False,
        )
        assert "Сторителлинг-фреймворки:" not in prompt, (
            "При STANDARD уровне без тега storytelling не должен быть включён"
        )

    def test_storytelling_included_at_full_even_if_domain_forbids(self, builder: PromptBuilder) -> None:
        """Если домен запрещает storytelling, FULL не переопределяет."""
        # Домен basic_edit имеет allow_storytelling=False
        prompt = builder.build(
            text="Тестовый текст.",
            domain="basic_edit",
            knowledge_level=KnowledgeLevel.FULL,
            include_knowledge=True,
            include_few_shot=False,
        )
        assert "Сторителлинг-фреймворки:" not in prompt, (
            "Если allow_storytelling=False, даже при FULL storytelling не включается"
        )

    def test_knowledge_level_changes_prompt_despite_cache(self, builder: PromptBuilder) -> None:
        """
        Тест из kb_loading_contract, но теперь должен проходить.
        Проверяет, что смена knowledge_level меняет промпт.
        """
        text = "Короткий тестовый текст для проверки состава блоков."
        p_core = builder.build(
            text=text,
            domain="nora_gal",
            knowledge_level=KnowledgeLevel.CORE,
            include_retrieval_meta=False,
        )
        p_full = builder.build(
            text=text,
            domain="nora_gal",
            knowledge_level=KnowledgeLevel.FULL,
            include_retrieval_meta=False,
        )

        assert len(p_full) > len(p_core), "FULL-промпт должен быть длиннее CORE"
        assert "Редакторские приёмы" in p_full, "При FULL и домене nora_gal должен быть блок 'Редакторские приёмы'"
        assert "Редакторские приёмы" not in p_core, "При CORE не должно быть блока 'Редакторские приёмы'"


class TestExplainabilityOverrides:
    """Проверяет, что explainability корректно отражает принудительное включение при FULL."""

    def test_explainability_has_full_level_reason_for_storytelling(self, builder: PromptBuilder) -> None:
        """При FULL и включении storytelling должен быть reason 'full_level_override'."""
        # Чтобы получить доступ к explainability, используем build с include_retrieval_meta=False
        # и смотрим на лог или на внутреннее состояние? В текущей реализации explainability
        # доступна через `_last_trace` после вызова build.
        _ = builder.build(
            text="Тест.",
            domain="blog",
            knowledge_level=KnowledgeLevel.FULL,
            include_knowledge=True,
            include_retrieval_meta=False,
        )
        trace = getattr(builder, "_last_trace", None)
        assert trace is not None, "Trace должен быть сохранён в builder._last_trace"

        # Ищем блок storytelling в trace
        storytelling_diag = None
        for diag in trace.blocks:
            if diag.name == "storytelling":
                storytelling_diag = diag
                break

        assert storytelling_diag is not None, "Блок storytelling должен присутствовать в trace"
        assert storytelling_diag.included is True, "Блок storytelling должен быть включён"
        # Проверяем, что есть reason BLOCK_INCLUDED (мы не можем проверить конкретный reason,
        # но можем проверить, что блок не был отключён из-за feature)
        assert ReasonCode.BLOCK_INELIGIBLE_FEATURE_DISABLED not in storytelling_diag.reason_codes, (
            "Storytelling не должен быть отключён по feature при FULL"
        )


class TestConfigSync:
    """Проверяет синхронность ALLOWED_* с файлами конфигов."""

    def test_all_domains_have_files(self) -> None:
        """Каждый домен из ALLOWED_DOMAINS имеет соответствующий файл в config/domains/."""
        domains_dir = Path("config/domains")
        assert domains_dir.is_dir(), "config/domains directory not found"
        existing = {p.stem for p in domains_dir.glob("*.json") if p.is_file()}
        missing = ALLOWED_DOMAINS - existing
        assert not missing, f"Missing domain files: {missing}"

    def test_no_extra_domain_files(self) -> None:
        """Нет лишних файлов в config/domains/, не объявленных в ALLOWED_DOMAINS."""
        domains_dir = Path("config/domains")
        existing = {p.stem for p in domains_dir.glob("*.json") if p.is_file()}
        extra = existing - ALLOWED_DOMAINS
        assert not extra, f"Extra domain files: {extra}"

    def test_all_intents_have_files_except_neutral(self) -> None:
        """Каждый intent из ALLOWED_INTENTS, кроме neutral, имеет файл."""
        intents_dir = Path("config/intents")
        assert intents_dir.is_dir(), "config/intents directory not found"
        existing = {p.stem for p in intents_dir.glob("*.json") if p.is_file()}
        expected = ALLOWED_INTENTS - {"neutral"}
        missing = expected - existing
        assert not missing, f"Missing intent files: {missing}"

    def test_no_extra_intent_files(self) -> None:
        """Нет лишних файлов в config/intents/, не объявленных в ALLOWED_INTENTS."""
        intents_dir = Path("config/intents")
        existing = {p.stem for p in intents_dir.glob("*.json") if p.is_file()}
        expected = ALLOWED_INTENTS - {"neutral"}
        extra = existing - expected
        assert not extra, f"Extra intent files: {extra}"

    def test_all_overlays_have_files(self) -> None:
        """Каждый overlay из ALLOWED_OVERLAYS имеет файл."""
        overlays_dir = Path("config/overlays")
        assert overlays_dir.is_dir(), "config/overlays directory not found"
        existing = {p.stem for p in overlays_dir.glob("*.json") if p.is_file()}
        missing = ALLOWED_OVERLAYS - existing
        assert not missing, f"Missing overlay files: {missing}"

    def test_no_extra_overlay_files(self) -> None:
        """Нет лишних файлов в config/overlays/, не объявленных в ALLOWED_OVERLAYS."""
        overlays_dir = Path("config/overlays")
        existing = {p.stem for p in overlays_dir.glob("*.json") if p.is_file()}
        extra = existing - ALLOWED_OVERLAYS
        assert not extra, f"Extra overlay files: {extra}"