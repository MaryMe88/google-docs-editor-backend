# tests/test_kb_loading_contract.py
"""
Контрактные тесты загрузки базы знаний и формирования блоков промпта.
Фиксируют исправления BUG-1, BUG-2, BUG-3, BUG-6.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

# Перезагружаем модуль, чтобы сбросить любые глобальные моки
import src.prompt_builder
importlib.reload(src.prompt_builder)

from src.prompt_builder import load_knowledge_base, PromptBuilder
from src.config_types import KnowledgeLevel
from src.knowledge_retrieval import _make_dedupe_key
from tests.conftest import KB_PATH   # <-- FIX: импорт из conftest

CONFIG_PATH = Path("config")


@pytest.fixture(autouse=True)
def reload_module():
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
        domain="nora_gal",
        knowledge_level=KnowledgeLevel.CORE,
        include_retrieval_meta=False,
    )
    p_full = pb.build(
        text=text,
        domain="nora_gal",
        knowledge_level=KnowledgeLevel.FULL,
        include_retrieval_meta=False,
    )

    assert len(p_full) > len(p_core), "FULL-промпт должен быть длиннее CORE"
    assert "Редакторские приёмы" in p_full, "При FULL и домене nora_gal должен быть блок 'Редакторские приёмы'"
    assert "Редакторские приёмы" not in p_core, "При CORE не должно быть блока 'Редакторские приёмы'"


def test_dedupe_keeps_distinct_structural():
    a = {"name": "X", "description": "D", "steps": [{"name": "s1", "description": "a"}]}
    b = {"name": "X", "description": "D", "steps": [{"name": "s2", "description": "b"}]}
    assert _make_dedupe_key(a) != _make_dedupe_key(b)


# NEW: Тесты для пилотного реорганизации
def test_case_study_json_contract():
    """Проверяет структуру и загрузку нового файла case_study.json."""
    from src.kb_manifest_loader import load_manifest

    # 1. Проверка наличия файла и валидности JSON
    path = KB_PATH / "genres" / "business" / "case_study.json"
    assert path.exists(), f"Файл {path} не найден"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert "tags" in data and "casestudy" in data["tags"]
    assert "templates" in data
    assert len(data["templates"]) >= 5, "Должно быть минимум 5 записей"

    for tmpl in data["templates"]:
        assert "id" in tmpl
        assert "name" in tmpl
        assert "sections" in tmpl
        assert isinstance(tmpl["sections"], list)
        assert len(tmpl["sections"]) > 0

    # 2. Проверка загрузки через манифест с тегом casestudy
    manifest = load_manifest(KB_PATH / "kb_manifest.json")
    entries = [e for e in manifest if e.file == "genres/business/case_study.json"]
    assert len(entries) == 1, "Манифест должен содержать ровно одну запись для case_study.json"
    entry = entries[0]
    assert entry.load_mode == "by_tags"
    assert "casestudy" in entry.tags
    assert entry.block_name == "case_study_templates", (
        "Жанровый файл кейса должен грузиться в собственный блок, а не в storytelling"
    )

    kb = load_knowledge_base(KB_PATH, active_tags={"casestudy"}, load_all=False)
    block = kb.get("case_study_templates")
    assert block is not None, "Блок case_study_templates не загружен"
    ids = [rec.get("id") for rec in block if isinstance(rec, dict)]
    assert "case_study_composition" in ids, "Запись 'case_study_composition' не найдена"
    results_records = [rec for rec in block if "results" in rec.get("tags", [])]
    assert len(results_records) >= 1, "Нет записей с тегом 'results'"