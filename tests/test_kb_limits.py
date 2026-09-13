from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

from src.prompt_builder import (
    KB_LIMIT_MAX,
    KB_LIMIT_MIN,
    DomainConfig,
    KnowledgeBlockRequest,
    LimitsConfig,
    PromptBuilder,
    load_domain_config,
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
    m = _merged(
        pb,
        {
            "grammar_candidates": 20,
            "stop_words_items": 3,
            "logic_candidates": 15,
        },
    )
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

    dc = load_domain_config("basic_edit", config_root)

    assert "grammer" not in dc.kb_limits
    assert dc.kb_limits.get("grammar") == 4

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

    warnings_found = [rec for rec in caplog.records if "вне диапазона" in rec.message]
    # значение 0 допустимо, поэтому предупреждений только для style и logic
    assert len(warnings_found) >= 2


def test_load_domain_config_bool_value_rejected(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    config_root = _write_domain(tmp_path, "basic_edit", {"grammar": True})
    dc = load_domain_config("basic_edit", config_root)

    assert "grammar" not in dc.kb_limits
    assert any("не число" in rec.message for rec in caplog.records)


# ------------------------------------------------------------------
# Проверяем наличие поля limits в KnowledgeBlockRequest
# ------------------------------------------------------------------
def test_build_knowledge_block_accepts_limits_param():
    """Проверяем, что KnowledgeBlockRequest имеет поле limits."""
    import dataclasses

    from src.prompt_builder import KnowledgeBlockRequest

    fields = {f.name for f in dataclasses.fields(KnowledgeBlockRequest)}
    assert "limits" in fields, "KnowledgeBlockRequest должен иметь поле 'limits'"


# ------------------------------------------------------------------
# Проверяем, что внутри _build_knowledge_block эффективные лимиты
# применяются к стоп-словам и к вызову _process_kb_block.
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
        mock_kb.get.side_effect = lambda key, default=None: {
            "stop_words": {"cat": ["w1", "w2"]},
            "composition_principles": [],
        }.get(key, default)

        mock_cache.get_or_load_multi.return_value = mock_kb

        # Подменяем _process_kb_block, чтобы проверить переданные лимиты.
        # Патчим в модуле builder, где имя реально используется (там оно
        # импортировано через from ._patchable import _process_kb_block).
        with patch("src.prompt_builder.builder._process_kb_block") as mock_process:
            # Создаём KnowledgeBlockRequest с кастомными лимитами
            custom_limits = LimitsConfig(stop_words_category=2, stop_words_items=1)
            req = KnowledgeBlockRequest(
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
                storytelling_enabled=True,
                marketing_enabled=True,
                antiai_enabled=False,
                rhetoric_enabled=False,
                nkrj_enabled=False,
                editorial_enabled=False,
                return_trace=False,
                semantic_rerank=False,
            )
            # Вызываем _build_knowledge_block с request
            _, _, _ = pb._build_knowledge_block(req)

            # Проверяем, что _process_kb_block вызван с limits=custom_limits.
            # _process_kb_block вызывается с именованными аргументами
            # (config=..., ctx=...), поэтому ctx лежит в kwargs, а не в args.
            for call in mock_process.call_args_list:
                ctx = call.kwargs["ctx"]
                assert ctx.limits is custom_limits
