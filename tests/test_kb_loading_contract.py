"""
Контрактные тесты загрузки базы знаний и формирования блоков промпта.
Фиксируют исправления BUG-1, BUG-2, BUG-3, BUG-6.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.prompt_builder import PromptBuilder, load_knowledge_base
from src.config_types import KnowledgeLevel

KB_PATH = Path("knowledge_base")
CONFIG_PATH = Path("config")


@pytest.fixture()
def kb_all():
    return load_knowledge_base(KB_PATH, active_tags=set(), intent=None, load_all=True)


def test_all_manifest_blocks_are_non_empty(kb_all):
    """BUG-1: ранее выпадавшие блоки теперь присутствуют и непусты."""
    for block in (
        "grammar_errors",
        "stylistic_issues",
        "composition_principles",
        "composition_errors",
        "local_cohesion",
        "stop_words",
        "nkrj_structure_patterns",
    ):
        value = kb_all.get(block)
        assert value, f"KB block '{block}' пуст или отсутствует после загрузки"


def test_dict_blocks_are_dicts(kb_all):
    """BUG-2: словарные блоки загружаются именно как dict, не как list."""
    for block in ("stop_words", "nkrj_structure_patterns"):
        assert isinstance(kb_all.get(block), dict), f"'{block}' должен быть dict"


def test_list_blocks_are_lists(kb_all):
    """Списочные блоки остаются списками записей."""
    for block in ("grammar_errors", "stylistic_issues", "composition_principles"):
        assert isinstance(kb_all.get(block), list), f"'{block}' должен быть list"


def test_stop_words_section_appears_in_prompt():
    """BUG-1/BUG-2: секция стоп-слов реально попадает в промпт."""
    pb = PromptBuilder(config_path=CONFIG_PATH, kb_path=KB_PATH)
    pb.startup_check()
    prompt = pb.build(
        text="В настоящий момент компания оказывает помощь. Уникальный лучший продукт, срочно закажите!",
        domain="basic_edit",
        knowledge_level=KnowledgeLevel.FULL,
    )
    assert "Стоп-слова" in prompt


def test_knowledge_level_changes_prompt_despite_cache():
    """
    BUG-3: разный knowledge_level даёт разный промпт (нет залипания кеша).
    Используем домен fiction, у которого есть primary-тег storytelling,
    что гарантирует загрузку соответствующего KB-блока при FULL.
    """
    pb = PromptBuilder(config_path=CONFIG_PATH, kb_path=KB_PATH)
    pb.startup_check()
    text = "Короткий тестовый текст для проверки состава блоков."

    p_core = pb.build(
        text=text,
        domain="fiction",
        knowledge_level=KnowledgeLevel.CORE,
        include_retrieval_meta=False,
    )
    p_full = pb.build(
        text=text,
        domain="fiction",
        knowledge_level=KnowledgeLevel.FULL,
        include_retrieval_meta=False,
    )

    # Проверяем, что FULL-промпт длиннее CORE
    assert len(p_full) > len(p_core), (
        "FULL-промпт должен быть длиннее CORE, так как добавляются storytelling, rhetoric и др."
    )

    # Проверяем наличие заголовка storytelling в FULL и его отсутствие в CORE
    assert "Сторителлинг-фреймворки" in p_full
    assert "Сторителлинг-фреймворки" not in p_core


def test_dedupe_keeps_distinct_structural():
    """BUG-6: структурные записи с одинаковыми name/description, но разными steps, не схлопываются."""
    from src.knowledge_retrieval import _make_dedupe_key

    a = {"name": "X", "description": "D", "steps": [{"name": "s1", "description": "a"}]}
    b = {"name": "X", "description": "D", "steps": [{"name": "s2", "description": "b"}]}
    assert _make_dedupe_key(a) != _make_dedupe_key(b)