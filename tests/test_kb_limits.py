from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.prompt_builder import (
    PromptBuilder,
    load_domain_config,
    LimitsConfig,
    DomainConfig,
    KB_LIMIT_MIN,
    KB_LIMIT_MAX,
)


def _merged(pb: PromptBuilder, kb_limits: dict) -> LimitsConfig:
    """Хелпер: вызывает _merge_domain_limits с временным DomainConfig."""
    return pb._merge_domain_limits(
        DomainConfig(
            name="probe",
            system_rules="",
            tone="neutral",
            kb_limits=kb_limits,
        )
    )


def test_merge_overrides_top_level_limits():
    pb = PromptBuilder()
    m = _merged(pb, {"grammar": 3, "style": 2})
    assert m.grammar == 3
    assert m.style == 2
    assert m.logic == pb._limits.logic  # не переопределён


def test_merge_overrides_candidates_and_stop_words_items():
    pb = PromptBuilder()
    m = _merged(pb, {
        "grammar_candidates": 20,
        "stop_words_items": 3,
        "logic_candidates": 15,
    })
    assert m.grammar_candidates == 20
    assert m.stop_words_items == 3
    assert m.logic_candidates == 15


def test_merge_cohesion_alias():
    pb = PromptBuilder()
    # только cohesion
    m1 = _merged(pb, {"cohesion": 2})
    assert m1.cohesion == 2

    # только local_cohesion
    m2 = _merged(pb, {"local_cohesion": 7})
    assert m2.cohesion == 7

    # оба — приоритет у cohesion
    m3 = _merged(pb, {"cohesion": 2, "local_cohesion": 9})
    assert m3.cohesion == 2


def test_build_knowledge_block_accepts_limits_param():
    """Проверяем, что сигнатура _build_knowledge_block содержит параметр limits."""
    import inspect
    sig = inspect.signature(PromptBuilder._build_knowledge_block)
    assert "limits" in sig.parameters
    assert sig.parameters["limits"].default is None


# ------------------------------------------------------------------
# Вспомогательная функция для создания временного домена
# ------------------------------------------------------------------
def _write_domain(tmp_path: Path, domain_name: str, kb_limits: dict) -> Path:
    """Создаёт временный файл домена и возвращает корень config."""
    config_root = tmp_path / "config"
    domains_dir = config_root / "domains"
    domains_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "name": domain_name,
        "system_rules": "x",
        "tone": "neutral",
        "kb_limits": kb_limits,
    }
    (domains_dir / f"{domain_name}.json").write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )
    return config_root


def test_load_domain_config_unknown_key_ignored_with_warning(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    config_root = _write_domain(tmp_path, "basic_edit", {"grammer": 5, "grammar": 4})

    # сигнатура load_domain_config: (domain, base_path=Path("config"))
    dc = load_domain_config("basic_edit", config_root)

    assert "grammer" not in dc.kb_limits
    assert dc.kb_limits.get("grammar") == 4

    # Проверяем, что было предупреждение о grammer
    assert any("grammer" in rec.message for rec in caplog.records)


def test_load_domain_config_range_clamped(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    config_root = _write_domain(
        tmp_path,
        "basic_edit",
        {"grammar": 0, "style": -3, "logic": 100000},
    )
    dc = load_domain_config("basic_edit", config_root)

    assert dc.kb_limits["grammar"] == KB_LIMIT_MIN
    assert dc.kb_limits["style"] == KB_LIMIT_MIN
    assert dc.kb_limits["logic"] == KB_LIMIT_MAX

    # Проверяем, что были предупреждения о зажиме
    warnings_found = [rec for rec in caplog.records if "вне диапазона" in rec.message]
    assert len(warnings_found) >= 3


def test_load_domain_config_bool_value_rejected(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    config_root = _write_domain(tmp_path, "basic_edit", {"grammar": True})
    dc = load_domain_config("basic_edit", config_root)

    assert "grammar" not in dc.kb_limits
    assert any("не число" in rec.message for rec in caplog.records)


# ------------------------------------------------------------------
# Интеграционный тест: _build_knowledge_block использует переданные лимиты
# ------------------------------------------------------------------
def test_build_knowledge_block_uses_passed_limits():
    """
    Проверяем, что внутри _build_knowledge_block эффективные лимиты
    применяются к стоп-словам и к вызову _process_kb_block.
    """
    pb = PromptBuilder()
    # Создаём мок для budget, чтобы получить stop_words_budget с enabled=True
    mock_budget = patch("src.prompt_builder.KnowledgeBudget").start()
    mock_budget.get.return_value = type(
        "BlockBudget", (), {"enabled": True, "entry_limit": 5}
    )()

    # Мокаем kb.get так, чтобы он принимал два аргумента
    with patch.object(pb, "_kb_cache") as mock_cache:
        mock_kb = patch("src.prompt_builder.KnowledgeBase").start()
        # Исправлено: lambda с default
        mock_kb.get.side_effect = lambda key, default=None: {
            "stop_words": {"cat": ["w1", "w2"]},
            "composition_principles": [],
        }.get(key, default)

        mock_cache.get_or_load_multi.return_value = mock_kb

        # Подменяем _process_kb_block, чтобы проверить переданные лимиты
        with patch("src.prompt_builder._process_kb_block") as mock_process:
            # Вызываем _build_knowledge_block с кастомными лимитами
            custom_limits = LimitsConfig(stop_words_category=2, stop_words_items=1)
            _, _, _ = pb._build_knowledge_block(
                text="test",
                primary_tags=set(),
                expanded_tags=set(),
                budget=mock_budget,
                domain="test",
                intent=None,
                overlays=[],
                include_few_shot=False,
                total_few_shot_used=0,
                few_shot_seed=None,
                limits=custom_limits,
            )

            # Проверяем, что _process_kb_block вызван с limits=custom_limits
            for call in mock_process.call_args_list:
                kwargs = call[1]
                assert kwargs.get("limits") is custom_limits