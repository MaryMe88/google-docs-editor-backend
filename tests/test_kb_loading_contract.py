# tests/test_kb_loading_contract.py
"""
Контрактные тесты загрузки базы знаний и формирования блоков промпта.
Фиксируют исправления BUG-1, BUG-2, BUG-3, BUG-6.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# Перезагружаем модуль, чтобы сбросить любые глобальные моки
import src.prompt_builder
importlib.reload(src.prompt_builder)

from src.prompt_builder import load_knowledge_base, PromptBuilder
from src.config_types import KnowledgeLevel
from src.knowledge_retrieval import _make_dedupe_key

KB_PATH = Path("knowledge_base")
CONFIG_PATH = Path("config")


@pytest.fixture(autouse=True)
def reload_module():
    """Перезагружаем модуль перед каждым тестом для защиты от протекающих моков."""
    importlib.reload(src.prompt_builder)
    yield


def test_all_manifest_blocks_are_non_empty():
    kb = load_knowledge_base(KB_PATH, load_all=True)
    for block in (
        "grammar_errors",
        "stylistic_issues",
        "composition_principles",
        "composition_errors",
        "local_cohesion",
        "stop_words",
        "nkrj_structure_patterns",
    ):
        value = kb.get(block)
        assert value, f"KB block '{block}' пуст или отсутствует после загрузки"


def test_dict_blocks_are_dicts():
    kb = load_knowledge_base(KB_PATH, load_all=True)
    for block in ("stop_words", "nkrj_structure_patterns"):
        actual = kb.get(block)
        assert isinstance(actual, dict), (
            f"'{block}' должен быть dict, получили {type(actual).__name__}"
        )


def test_list_blocks_are_lists():
    kb = load_knowledge_base(KB_PATH, load_all=True)
    for block in ("grammar_errors", "stylistic_issues", "composition_principles"):
        actual = kb.get(block)
        assert isinstance(actual, list), (
            f"'{block}' должен быть list, получили {type(actual).__name__}"
        )


def test_stop_words_section_appears_in_prompt():
    pb = PromptBuilder(config_path=CONFIG_PATH, kb_path=KB_PATH)
    pb.startup_check()
    prompt = pb.build(
        text="В настоящий момент компания оказывает помощь. Уникальный лучший продукт, срочно закажите!",
        domain="basic_edit",
        knowledge_level=KnowledgeLevel.FULL,
    )
    assert "Стоп-слова" in prompt


def test_knowledge_level_changes_prompt_despite_cache():
    pb = PromptBuilder(config_path=CONFIG_PATH, kb_path=KB_PATH)
    pb.startup_check()
    text = "Короткий тестовый текст для проверки состава блоков."

    p_core = pb.build(
        text=text,
        domain="basic_edit",
        knowledge_level=KnowledgeLevel.CORE,
        include_retrieval_meta=False,
    )
    p_full = pb.build(
        text=text,
        domain="basic_edit",
        knowledge_level=KnowledgeLevel.FULL,
        include_retrieval_meta=False,
    )

    assert len(p_full) > len(p_core), "FULL-промпт должен быть длиннее CORE"
    assert "Редакторские приёмы" in p_full, "При FULL и домене basic_edit должен быть блок 'Редакторские приёмы'"
    assert "Редакторские приёмы" not in p_core, "При CORE не должно быть блока 'Редакторские приёмы'"


def test_dedupe_keeps_distinct_structural():
    a = {"name": "X", "description": "D", "steps": [{"name": "s1", "description": "a"}]}
    b = {"name": "X", "description": "D", "steps": [{"name": "s2", "description": "b"}]}
    assert _make_dedupe_key(a) != _make_dedupe_key(b)