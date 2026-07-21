# tests/test_explainability.py
"""
Тесты для third-итерации: explainability и диагностика.
Обновлены для четвёртой итерации: используют registry и новые проверки startup_checks.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.prompt_builder import (
    PromptBuilder,
    resolve_prompt_features,
    load_domain_config,
    load_intent_config,
)
from src.config_types import (
    FeatureResolutionResult,
    AssemblyTrace,
    AssemblyBlockDiagnostics,
    KnowledgeLevel,
    KnowledgeBudget,
    BlockBudget,
    LimitsConfig,
    DomainConfig,
)
from src.reason_codes import ReasonCode, ACTIVATION_REASONS, SUPPRESSION_REASONS

# NEW: импорт registry для проверки алиасов
from src.registry import check_alias_consistency

# NEW: импорт проверок из startup_checks (они остались)
from src.startup_checks import (
    _check_feature_resolution_invariants,
    _check_assembly_diagnostics_invariants,
    _check_registry_consistency,
)


@pytest.fixture
def builder():
    return PromptBuilder(
        config_path=Path("config"),
        kb_path=Path("knowledge_base"),
    )


class TestFeatureResolutionExplainability:
    """Тесты для resolve_prompt_features и диагностических полей."""

    def test_resolve_prompt_features_returns_expected_structure(self, builder):
        domain = "blog"
        intent = None
        overlays = []
        domain_config = load_domain_config(domain, builder.config_path)
        intent_config = load_intent_config(intent, builder.config_path)
        overlay_configs = []

        result = resolve_prompt_features(
            domain=domain,
            intent=intent,
            overlays=overlays,
            domain_config=domain_config,
            intent_config=intent_config,
            overlay_configs=overlay_configs,
        )

        expected_keys = {
            "tags", "effective_intent", "effective_overlays", "suppressed_layers", "warnings",
            "storytelling_enabled", "marketing_enabled", "antiai_enabled", "rhetoric_enabled",
            "nkrj_enabled", "editorial_enabled",
            "activated_features", "suppressed_features",
            "activation_reasons", "suppression_reasons",
            "recognized_aliases", "ignored_unknown_values",
        }
        assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - result.keys()}"

    def test_resolve_prompt_features_storytelling_activation_reasons(self, builder):
        domain = "blog"
        intent = "storytelling"
        overlays = []
        domain_config = load_domain_config(domain, builder.config_path)
        intent_config = load_intent_config(intent, builder.config_path)
        overlay_configs = []

        result = resolve_prompt_features(
            domain=domain,
            intent=intent,
            overlays=overlays,
            domain_config=domain_config,
            intent_config=intent_config,
            overlay_configs=overlay_configs,
        )

        assert result["storytelling_enabled"] is True
        assert "storytelling" in result["activation_reasons"]
        reasons = result["activation_reasons"]["storytelling"]
        assert ReasonCode.DOMAIN_ALLOWS_STORYTELLING in reasons
        assert ReasonCode.RECOGNIZED_STORYTELLING_ALIAS in reasons
        assert "storytelling" in result["recognized_aliases"]
        assert "storytelling" in result["recognized_aliases"]["storytelling"]

    def test_resolve_prompt_features_marketing_suppression_when_disabled(self, builder):
        domain = "deai"
        intent = "marketingpush"
        overlays = []
        domain_config = load_domain_config(domain, builder.config_path)
        intent_config = load_intent_config(intent, builder.config_path)
        overlay_configs = []

        result = resolve_prompt_features(
            domain=domain,
            intent=intent,
            overlays=overlays,
            domain_config=domain_config,
            intent_config=intent_config,
            overlay_configs=overlay_configs,
        )

        assert result["marketing_enabled"] is False
        assert "marketing" in result["suppression_reasons"]
        reasons = result["suppression_reasons"]["marketing"]
        assert ReasonCode.DOMAIN_DENIES_MARKETING in reasons

    def test_resolve_prompt_features_ignored_unknown_intent(self, builder):
        domain = "blog"
        intent = "nonexistent_intent"
        overlays = []
        domain_config = load_domain_config(domain, builder.config_path)
        intent_config = load_intent_config(intent, builder.config_path)
        overlay_configs = []

        result = resolve_prompt_features(
            domain=domain,
            intent=intent,
            overlays=overlays,
            domain_config=domain_config,
            intent_config=intent_config,
            overlay_configs=overlay_configs,
        )

        assert "nonexistent_intent" in result["ignored_unknown_values"]
        assert result["effective_intent"] is None

    def test_resolve_prompt_features_ignored_unknown_overlay(self, builder):
        domain = "blog"
        intent = None
        overlays = ["unknown_overlay"]
        domain_config = load_domain_config(domain, builder.config_path)
        intent_config = load_intent_config(intent, builder.config_path)
        overlay_configs = []

        result = resolve_prompt_features(
            domain=domain,
            intent=intent,
            overlays=overlays,
            domain_config=domain_config,
            intent_config=intent_config,
            overlay_configs=overlay_configs,
        )

        assert "unknown_overlay" in result["ignored_unknown_values"]
        assert "unknown_overlay" not in result["effective_overlays"]

    def test_resolve_prompt_features_recognized_aliases_for_antiai(self, builder):
        domain = "deai"
        intent = None
        overlays = []
        domain_config = load_domain_config(domain, builder.config_path)
        intent_config = load_intent_config(intent, builder.config_path)
        overlay_configs = []

        result = resolve_prompt_features(
            domain=domain,
            intent=intent,
            overlays=overlays,
            domain_config=domain_config,
            intent_config=intent_config,
            overlay_configs=overlay_configs,
        )

        assert result["antiai_enabled"] is True
        assert "antiai" in result["activation_reasons"]
        assert "antiai" in result["recognized_aliases"]
        assert "deai" in result["recognized_aliases"]["antiai"]

    def test_resolve_prompt_features_suppression_by_incompatible_intent(self):
        from src.config_types import DomainConfig
        mock_domain = DomainConfig(
            name="blog",
            system_rules="",
            tone="neutral",
            allow_storytelling=True,
            allow_marketing=True,
            incompatible_intents=("marketingpush",),
        )
        intent = "marketingpush"
        overlays = []
        domain_config = mock_domain
        intent_config = None
        overlay_configs = []

        result = resolve_prompt_features(
            domain="blog",
            intent=intent,
            overlays=overlays,
            domain_config=domain_config,
            intent_config=intent_config,
            overlay_configs=overlay_configs,
        )

        assert result["effective_intent"] is None
        assert "intent" in result["suppression_reasons"]
        assert ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_INTENT in result["suppression_reasons"]["intent"]


class TestAssemblyDiagnostics:
    """Тесты для диагностики сборки knowledge blocks."""

    def test_build_knowledge_block_with_diagnostics_returns_trace(self, builder):
        budget = KnowledgeBudget({
            "grammar": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "style": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "logic": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
            "composition": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
            "composition_errors": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
            "cohesion": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
            "storytelling": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
            "marketing": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
            "rhetoric": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
            "editorial": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
            "glossary": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
            "stop_words": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
            "nkrj": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
        })
        text = "Тест."
        primary_tags = {"grammar", "style"}
        expanded_tags = set()
        domain = "blog"
        intent = None
        overlays = []

        result = builder._build_knowledge_block(
            text=text,
            primary_tags=primary_tags,
            expanded_tags=expanded_tags,
            budget=budget,
            domain=domain,
            intent=intent,
            overlays=overlays,
            include_few_shot=False,
            total_few_shot_used=0,
            few_shot_seed=None,
            limits=LimitsConfig(),
            storytelling_enabled=False,
            marketing_enabled=False,
            antiai_enabled=False,
            rhetoric_enabled=False,
            nkrj_enabled=False,
            editorial_enabled=False,
            return_trace=True,
        )

        assert len(result) == 4
        knowledge_text, meta, total_used, trace = result
        assert isinstance(knowledge_text, str)
        assert isinstance(meta, dict)
        assert isinstance(total_used, int)
        assert isinstance(trace, AssemblyTrace)

    def test_build_knowledge_block_diagnostics_block_fields(self, builder):
        effective_limits = LimitsConfig()
        budget = KnowledgeBudget({
            "grammar": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "style": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "logic": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "composition": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "composition_errors": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "cohesion": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "storytelling": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "marketing": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "rhetoric": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "editorial": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "glossary": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "stop_words": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "nkrj": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
        })
        text = "Короткий текст для теста."
        primary_tags = {"grammar", "style", "editing", "clarity"}
        expanded_tags = set()
        domain = "blog"
        intent = None
        overlays = []

        _, _, _, trace = builder._build_knowledge_block(
            text=text,
            primary_tags=primary_tags,
            expanded_tags=expanded_tags,
            budget=budget,
            domain=domain,
            intent=intent,
            overlays=overlays,
            include_few_shot=False,
            total_few_shot_used=0,
            few_shot_seed=None,
            limits=effective_limits,
            storytelling_enabled=True,
            marketing_enabled=True,
            antiai_enabled=True,
            rhetoric_enabled=True,
            nkrj_enabled=True,
            editorial_enabled=True,
            return_trace=True,
        )

        assert len(trace.blocks) > 0
        for diag in trace.blocks:
            assert hasattr(diag, 'name')
            assert hasattr(diag, 'eligible')
            assert hasattr(diag, 'included')
            assert hasattr(diag, 'reason_codes')
            assert hasattr(diag, 'empty')
            assert hasattr(diag, 'char_count')
            assert hasattr(diag, 'entries_count')
            assert isinstance(diag.reason_codes, list)
            for code in diag.reason_codes:
                assert isinstance(code, str)

    def test_build_knowledge_block_feature_gated_blocks_not_included_when_disabled(self, builder):
        budget = KnowledgeBudget({
            "storytelling": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "marketing": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "rhetoric": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "editorial": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
            "nkrj": BlockBudget(entry_limit=5, char_budget=None, enabled=True),
        })
        text = "Тест."
        primary_tags = {"storytelling", "marketing", "rhetoric", "editorial", "nkrj"}
        expanded_tags = set()
        domain = "blog"
        intent = None
        overlays = []

        _, _, _, trace = builder._build_knowledge_block(
            text=text,
            primary_tags=primary_tags,
            expanded_tags=expanded_tags,
            budget=budget,
            domain=domain,
            intent=intent,
            overlays=overlays,
            include_few_shot=False,
            total_few_shot_used=0,
            few_shot_seed=None,
            limits=LimitsConfig(),
            storytelling_enabled=False,
            marketing_enabled=False,
            antiai_enabled=False,
            rhetoric_enabled=False,
            nkrj_enabled=False,
            editorial_enabled=False,
            return_trace=True,
        )

        gated_block_names = {"storytelling", "marketing", "rhetoric", "editorial", "nkrj"}
        for diag in trace.blocks:
            if diag.name in gated_block_names:
                assert diag.included is False
                assert ReasonCode.BLOCK_INELIGIBLE_FEATURE_DISABLED in diag.reason_codes

    def test_build_knowledge_block_ineligible_if_budget_disabled(self, builder):
        budget = KnowledgeBudget({
            "grammar": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
        })
        text = "Тест."
        primary_tags = {"grammar"}
        expanded_tags = set()
        domain = "blog"
        intent = None
        overlays = []

        _, _, _, trace = builder._build_knowledge_block(
            text=text,
            primary_tags=primary_tags,
            expanded_tags=expanded_tags,
            budget=budget,
            domain=domain,
            intent=intent,
            overlays=overlays,
            include_few_shot=False,
            total_few_shot_used=0,
            few_shot_seed=None,
            limits=LimitsConfig(),
            storytelling_enabled=False,
            marketing_enabled=False,
            antiai_enabled=False,
            rhetoric_enabled=False,
            nkrj_enabled=False,
            editorial_enabled=False,
            return_trace=True,
        )

        grammar_diag = next((d for d in trace.blocks if d.name == "grammar"), None)
        assert grammar_diag is not None
        assert grammar_diag.eligible is False
        assert grammar_diag.included is False
        assert ReasonCode.BLOCK_INELIGIBLE_BUDGET_DISABLED in grammar_diag.reason_codes


class TestStartupChecksExplainability:
    """Тесты для новых проверок в startup_checks.py (четвёртая итерация)."""

    def test_check_alias_consistency_does_not_raise(self, builder):
        """Проверяем, что check_alias_consistency из registry не выбрасывает исключений."""
        try:
            warnings = check_alias_consistency()
            # Просто убеждаемся, что это список
            assert isinstance(warnings, list)
        except Exception as e:
            pytest.fail(f"check_alias_consistency raised {e}")

    def test_check_registry_consistency_does_not_raise(self, builder):
        """Проверяем, что _check_registry_consistency не выбрасывает исключений."""
        try:
            _check_registry_consistency()
        except Exception as e:
            pytest.fail(f"_check_registry_consistency raised {e}")

    def test_check_feature_resolution_invariants_does_not_raise(self, builder):
        try:
            _check_feature_resolution_invariants(builder.config_path)
        except Exception as e:
            pytest.fail(f"_check_feature_resolution_invariants raised {e}")

    def test_check_assembly_diagnostics_invariants_does_not_raise(self, builder):
        try:
            _check_assembly_diagnostics_invariants(builder.config_path)
        except Exception as e:
            pytest.fail(f"_check_assembly_diagnostics_invariants raised {e}")