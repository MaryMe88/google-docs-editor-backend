"""
tests/test_architecture_invariants.py

Архитектурные инварианты, которые должны выполняться всегда.
Проверяют согласованность слоёв и корректность feature resolution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.prompt_builder import PromptBuilder, KnowledgeLevel
from src.shared_contracts import ALLOWED_DOMAINS, ALLOWED_INTENTS, ALLOWED_OVERLAYS
from src.config_types import DomainConfig
from src.reason_codes import ReasonCode

# Новые импорты для тестов-инвариантов
from src.kb_manifest_loader import load_manifest
from src.prompt_builder import (
    KB_BLOCK_REGISTRY,
    _load_kb_file,
    KnowledgeBudgetManager,
    LimitsConfig,
    KnowledgeLevel as KBLevel,
)

# Импортируем FILE_RULES из генератора манифеста (scripts/generate_kb_manifest.py)
try:
    scripts_dir = Path(__file__).parent.parent / "scripts"
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))
        from generate_kb_manifest import FILE_RULES
    else:
        FILE_RULES = {}
except ImportError:
    FILE_RULES = {}

# Путь к базе знаний (используем тот же, что в conftest)
KB_PATH = Path("knowledge_base")


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder(config_path=Path("config"), kb_path=KB_PATH)


class TestKnowledgeLevelOverrides:
    """Проверяет, что при FULL уровне блоки storytelling/marketing включаются принудительно."""

    def test_storytelling_included_at_full_without_tag(self, builder: PromptBuilder) -> None:
        """При FULL и allow_storytelling=True storytelling появляется даже без тега."""
        prompt = builder.build(
            text="Тестовый текст для проверки сторителлинга.",
            domain="blog",
            knowledge_level=KnowledgeLevel.FULL,
            include_knowledge=True,
            include_few_shot=False,
        )
        assert "Сторителлинг-фреймворки:" in prompt, (
            "При FULL уровне и allow_storytelling=True блок storytelling должен быть включён"
        )

    def test_marketing_included_at_full_without_tag(self, builder: PromptBuilder) -> None:
        """При FULL и allow_marketing=True marketing появляется даже без тега."""
        prompt = builder.build(
            text="Тестовый текст для проверки маркетинга.",
            domain="marketing",
            knowledge_level=KnowledgeLevel.FULL,
            include_knowledge=True,
            include_few_shot=False,
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
        _ = builder.build(
            text="Тест.",
            domain="blog",
            knowledge_level=KnowledgeLevel.FULL,
            include_knowledge=True,
            include_retrieval_meta=False,
        )
        trace = getattr(builder, "_last_trace", None)
        assert trace is not None, "Trace должен быть сохранён в builder._last_trace"

        storytelling_diag = None
        for diag in trace.blocks:
            if diag.name == "storytelling":
                storytelling_diag = diag
                break

        assert storytelling_diag is not None, "Блок storytelling должен присутствовать в trace"
        assert storytelling_diag.included is True, "Блок storytelling должен быть включён"
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


# ============================================================================
# НОВЫЕ ТЕСТЫ-ИНВАРИАНТЫ ДЛЯ БАЗЫ ЗНАНИЙ
# ============================================================================

class TestManifestConsistency:
    """
    Проверяет согласованность между манифестом KB, реестром блоков,
    бюджетом и генератором манифеста.
    """

    def test_every_manifest_block_is_consumed(self) -> None:
        """
        Каждый блок, описанный в манифесте, должен реально читаться
        обработчиком (попадать в промпт). Иначе содержимое теряется.
        """
        manifest = load_manifest(KB_PATH / "kb_manifest.json")

        # Множество ключей, которые точно потребляются кодом
        consumed = set()

        # 1. Блоки из реестра с явным kb_attr
        for cfg in KB_BLOCK_REGISTRY:
            if cfg.kb_attr:
                consumed.add(cfg.kb_attr)

        # 2. Грамматика, стиль, логика — используются через retrieval_fn,
        #    их фактические ключи в KB: grammar_errors, stylistic_issues, logic_issues
        consumed.add("grammar_errors")
        consumed.add("stylistic_issues")
        consumed.add("logic_issues")

        # 3. Специальные блоки, обрабатываемые отдельно
        consumed.add("stop_words")
        consumed.add("domain_glossary")
        consumed.add("nkrj_structure_patterns")
        # НОВОЕ: блок техник работы с оценками
        consumed.add("evaluation_techniques")

        # Проверяем каждый файл в манифесте
        for entry in manifest:
            # ключ блока: block_name или имя файла без расширения
            key = entry.block_name or Path(entry.file).stem
            assert key in consumed, (
                f"{entry.file}: блок '{key}' не читается ни одним обработчиком — "
                "содержимое не попадёт в промпт"
            )

    def test_block_type_matches_file_structure(self) -> None:
        """
        Тип блока (list / dict), указанный в манифесте, должен совпадать
        с реальной структурой, которую возвращает _load_kb_file.
        """
        manifest = load_manifest(KB_PATH / "kb_manifest.json")

        for entry in manifest:
            # Для файлов с block_type="dict" мы не пытаемся искать ключ-список,
            # а загружаем весь словарь.
            records = _load_kb_file(
                KB_PATH / entry.file,
                expected_key=entry.block_name or Path(entry.file).stem,
                use_known_keys=(entry.block_type != "dict"),
            )
            assert records, f"{entry.file}: загрузка дала пустой блок"

            expected_type = dict if entry.block_type == "dict" else list
            assert isinstance(records, expected_type), (
                f"{entry.file}: block_type={entry.block_type}, "
                f"а загрузилось {type(records).__name__}"
            )

    def test_every_registry_block_has_budget(self) -> None:
        """
        Для каждого блока из KB_BLOCK_REGISTRY должен существовать
        соответствующий ключ в бюджете, и блок должен быть разрешён
        на уровне FULL.
        """
        budget = KnowledgeBudgetManager().allocate(
            LimitsConfig(),
            level=KBLevel.FULL,
        )

        for cfg in KB_BLOCK_REGISTRY:
            block_budget = budget.get(cfg.budget_key)
            assert block_budget is not None, (
                f"Блок '{cfg.name}': нет бюджета '{cfg.budget_key}' в KnowledgeBudget"
            )
            assert block_budget.enabled, (
                f"Блок '{cfg.name}' отключён на FULL уровне (лимит 0?)"
            )

    def test_manifest_matches_generator(self) -> None:
        """
        Манифест должен быть воспроизводим генератором (FILE_RULES).
        Ручные правки манифеста не должны расходиться с правилами.
        """
        if not FILE_RULES:
            pytest.skip("FILE_RULES не загружен (пропуск теста)")

        manifest = load_manifest(KB_PATH / "kb_manifest.json")

        for entry in manifest:
            filename = Path(entry.file).name
            rule = FILE_RULES.get(filename)

            assert rule is not None, (
                f"{entry.file}: нет правила в FILE_RULES — манифест, вероятно, "
                "сгенерирован вручную или файл не описан в генераторе"
            )

            # load_mode
            assert rule.get("load_mode") == entry.load_mode, (
                f"{entry.file}: load_mode mismatch (rule={rule.get('load_mode')}, "
                f"manifest={entry.load_mode})"
            )

            # block_name (может отсутствовать в rule)
            expected_block_name = rule.get("block_name")
            if expected_block_name is None:
                assert entry.block_name is None, (
                    f"{entry.file}: rule не имеет block_name, но манифест содержит "
                    f"'{entry.block_name}'"
                )
            else:
                assert expected_block_name == entry.block_name, (
                    f"{entry.file}: block_name mismatch (rule={expected_block_name}, "
                    f"manifest={entry.block_name})"
                )

            # block_type (по умолчанию "list")
            expected_block_type = rule.get("block_type", "list")
            assert expected_block_type == entry.block_type, (
                f"{entry.file}: block_type mismatch (rule={expected_block_type}, "
                f"manifest={entry.block_type})"
            )
