# tests/test_prompt_builder.py
from __future__ import annotations

import warnings
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from src.config_types import (
    KnowledgeLevel,
    KnowledgeBudget,
    BlockBudget,
    DomainConfig,
    LimitsConfig,
)
from src.prompt_builder import (
    PromptBuilder,
    _get_confidence_note,
    _derive_seed,
    _process_kb_block,
    KB_BLOCK_REGISTRY,
    KBBlockConfig,
    _append_rule_entries,
    load_output_format,
    load_domain_config,
    load_intent_config,
)
from src.knowledge_retrieval import FallbackStage, _collect_with_budget


# ============================================================================
# Вспомогательная функция для создания мока KB с поддержкой .get()
# ============================================================================

def make_mock_kb(data: dict) -> MagicMock:
    """
    Создаёт MagicMock, который ведёт себя как KnowledgeBase:
    - имеет все ключи из data как атрибуты (например, kb.grammar_errors)
    - поддерживает метод .get(key, default)
    """
    kb = MagicMock()
    # Добавляем атрибуты
    for key, value in data.items():
        setattr(kb, key, value)
    # Реализуем метод get
    def get_side_effect(key, default=None):
        return data.get(key, default)
    kb.get.side_effect = get_side_effect
    return kb


# ============================================================================
# Старые тесты (без изменений)
# ============================================================================

def test_build_returns_string(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Это тестовый текст для проверки PromptBuilder.",
        domain="blog",
    )
    assert isinstance(result, str)
    assert "Роль:" in result
    assert "Домен:" in result
    assert "Исходный текст:" in result
    assert "Это тестовый текст" in result


def test_include_knowledge_false_omits_knowledge_block(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Текст без KB блока.",
        domain="blog",
        include_knowledge=False,
    )
    assert "База знаний:" not in result


def test_knowledge_level_none_disables_knowledge_content(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Текст с knowledge_level none.",
        domain="blog",
        include_knowledge=True,
        knowledge_level=KnowledgeLevel.NONE,
    )
    assert "Исходный текст:" in result
    assert "База знаний:" not in result


def test_knowledge_level_core_or_standard_does_not_crash(builder: PromptBuilder) -> None:
    core_result = builder.build(
        text="Проверка уровня core.",
        domain="blog",
        include_knowledge=True,
        knowledge_level=KnowledgeLevel.CORE,
    )
    standard_result = builder.build(
        text="Проверка уровня standard.",
        domain="blog",
        include_knowledge=True,
        knowledge_level=KnowledgeLevel.STANDARD,
    )
    assert isinstance(core_result, str)
    assert isinstance(standard_result, str)
    assert "Исходный текст:" in core_result
    assert "Исходный текст:" in standard_result


def test_full_level_without_optional_configs_does_not_crash(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Нужно сделать текст логичнее, чище и убедительнее.",
        domain="marketing",
        include_knowledge=True,
        knowledge_level=KnowledgeLevel.FULL,
        token_budget=1200,
    )
    assert isinstance(result, str)
    assert "Роль:" in result
    assert "Домен:" in result
    assert "Исходный текст:" in result


def test_invalid_domain_raises_assertion_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Unknown domain"):
        builder.build(text="Текст.", domain="science")


def test_invalid_intent_raises_assertion_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Unknown intent"):
        builder.build(text="Текст.", domain="blog", intent="unknown_intent")


def test_invalid_overlay_raises_assertion_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Unknown overlay"):
        builder.build(text="Текст.", domain="blog", overlays=["unknown_overlay"])


def test_empty_text_raises_value_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Text must not be empty"):
        builder.build(text="   ", domain="blog")


@pytest.mark.parametrize(("domain", "intent"), [("blog", "neutral")])
def test_supported_domain_intent_combinations(builder: PromptBuilder, domain: str, intent: str) -> None:
    result = builder.build(
        text="Проверка допустимой комбинации домена и intent.",
        domain=domain,
        intent=intent,
        include_knowledge=False,
    )
    assert isinstance(result, str)
    assert "Домен:" in result


# ---------------------------------------------------------------------------
# Few-shot тесты (PR‑2)
# ---------------------------------------------------------------------------

def test_include_few_shot_false_omits_examples(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Он согласился согласно приказа начальника.",
        domain="blog",
        include_knowledge=True,
        include_few_shot=False,
    )
    assert "Примеры редактирования" not in result


def test_include_few_shot_true_does_not_crash(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Он согласился согласно приказа начальника.",
        domain="blog",
        include_knowledge=True,
        include_few_shot=True,
    )
    assert isinstance(result, str)
    assert "Исходный текст:" in result


def test_include_few_shot_without_knowledge_does_nothing(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Текст без знаний.",
        domain="blog",
        include_knowledge=False,
        include_few_shot=True,
    )
    assert "База знаний:" not in result
    assert "Примеры редактирования" not in result

    result2 = builder.build(
        text="Текст без знаний.",
        domain="blog",
        include_knowledge=False,
        include_few_shot=False,
    )
    assert result == result2


# ---------------------------------------------------------------------------
# Тесты для ТП-2 (квалификатор уверенности) — с подменой KB_BLOCK_REGISTRY
# ---------------------------------------------------------------------------

def test_get_confidence_note_returns_correct_strings() -> None:
    """Проверяем, что функция возвращает правильные строки для разных stage."""
    assert _get_confidence_note(FallbackStage.STRONG) == ""
    assert _get_confidence_note(FallbackStage.EMPTY) == ""

    text_only = _get_confidence_note(FallbackStage.TEXT_ONLY)
    assert "смысловому совпадению" in text_only

    tag_only = _get_confidence_note(FallbackStage.TAG_ONLY)
    assert "теме раздела" in tag_only

    neutral = _get_confidence_note(FallbackStage.NEUTRAL)
    assert "теме раздела" in neutral


def test_confidence_note_inserted_for_tag_only_stage(builder: PromptBuilder) -> None:
    """При stage=TAG_ONLY в промпте должен появиться квалификатор "теме раздела"."""
    fake_entry = {
        "wrong": "несовпадающий текст",
        "rule": "правило грамматики",
        "tags": ["grammar"],
    }
    mock_fn = MagicMock(return_value=([fake_entry], FallbackStage.TAG_ONLY, 0))

    test_config = KBBlockConfig(
        name="grammar",
        budget_key="grammar",
        retrieval_fn=mock_fn,
        append_fn=_append_rule_entries,
        title="Грамматические ориентиры:",
        kb_attr=None,
        uses_structural_call=False,
        candidate_attr="grammar_candidates",
    )
    test_registry = [test_config]

    with patch("src.prompt_builder.KB_BLOCK_REGISTRY", test_registry):
        result = builder.build(
            text="Тестовый текст без совпадений.",
            domain="blog",
            include_knowledge=True,
            include_few_shot=False,
            knowledge_level=KnowledgeLevel.FULL,
            token_budget=None,
        )

    assert "Грамматические ориентиры:" in result
    assert "теме раздела" in result
    mock_fn.assert_called_once()


def test_confidence_note_not_inserted_for_strong_stage(builder: PromptBuilder) -> None:
    """
    При stage=STRONG квалификатор не должен появляться.
    Используем запись с точным совпадением по wrong (без correct, чтобы это было правило).
    """
    data = {
        "grammar_errors": [
            {"wrong": "Тестовый текст", "rule": "правило грамматики", "tags": ["grammar"]}
        ],
        "stylistic_issues": [],
        "logic_issues": [],
        "composition_principles": [],
        "composition_errors": [],
        "local_cohesion": [],
        "storytelling_frameworks": [],
        "marketing_templates": [],
        "rhetoric_frameworks": [],
        "editorial_techniques": [],
        "stop_words": {},
        "domain_glossary": {},
        "nkrj_structure_patterns": {},
    }
    kb = make_mock_kb(data)

    with patch('src.prompt_builder.load_knowledge_base', return_value=kb):
        result = builder.build(
            text="Тестовый текст",
            domain="blog",
            include_knowledge=True,
            include_few_shot=False,
            knowledge_level=KnowledgeLevel.FULL,
            token_budget=None,
        )
        # Блок грамматики должен быть
        assert "Грамматические ориентиры:" in result
        # Квалификатор не должен появиться
        assert "теме раздела" not in result
        assert "смысловому совпадению" not in result


def test_confidence_note_not_inserted_when_no_knowledge(builder: PromptBuilder) -> None:
    """
    Если блоки знаний пусты, квалификатор не добавляется.
    """
    data = {
        "grammar_errors": [],
        "stylistic_issues": [],
        "logic_issues": [],
        "composition_principles": [],
        "composition_errors": [],
        "local_cohesion": [],
        "storytelling_frameworks": [],
        "marketing_templates": [],
        "rhetoric_frameworks": [],
        "editorial_techniques": [],
        "stop_words": {},
        "domain_glossary": {},
        "nkrj_structure_patterns": {},
    }
    kb = make_mock_kb(data)

    with patch('src.prompt_builder.load_knowledge_base', return_value=kb):
        result = builder.build(
            text="Тестовый текст.",
            domain="blog",
            include_knowledge=True,
            include_few_shot=False,
            knowledge_level=KnowledgeLevel.FULL,
            token_budget=None,
        )
        # В промпте не должно быть квалификаторов
        assert "теме раздела" not in result
        assert "смысловому совпадению" not in result
        # Блоков знаний быть не должно
        assert "Грамматические ориентиры:" not in result


def test_confidence_note_appears_only_once_per_block(builder: PromptBuilder) -> None:
    """
    Квалификатор добавляется ровно один раз для блока,
    даже если в блоке несколько записей.
    """
    fake_entries = [
        {"wrong": "ошибка1", "rule": "правило1", "tags": ["grammar"]},
        {"wrong": "ошибка2", "rule": "правило2", "tags": ["grammar"]},
    ]
    mock_fn = MagicMock(return_value=(fake_entries, FallbackStage.TAG_ONLY, 0))

    test_config = KBBlockConfig(
        name="grammar",
        budget_key="grammar",
        retrieval_fn=mock_fn,
        append_fn=_append_rule_entries,
        title="Грамматические ориентиры:",
        kb_attr=None,
        uses_structural_call=False,
        candidate_attr="grammar_candidates",
    )
    test_registry = [test_config]

    with patch("src.prompt_builder.KB_BLOCK_REGISTRY", test_registry):
        result = builder.build(
            text="Тестовый текст без совпадений.",
            domain="blog",
            include_knowledge=True,
            include_few_shot=False,
            knowledge_level=KnowledgeLevel.FULL,
            token_budget=None,
        )

    count = result.count("теме раздела")
    assert count == 1, f"Квалификатор должен встречаться 1 раз, найдено {count}"
    assert "ошибка1" in result
    assert "ошибка2" in result
    mock_fn.assert_called_once()


def test_confidence_note_position_before_rules(builder: PromptBuilder) -> None:
    """
    Проверяем, что квалификатор идёт перед заголовком "Грамматические ориентиры:".
    """
    fake_entry = {
        "wrong": "несовпадающий текст",
        "rule": "правило грамматики",
        "tags": ["grammar"],
    }
    mock_fn = MagicMock(return_value=([fake_entry], FallbackStage.TAG_ONLY, 0))

    test_config = KBBlockConfig(
        name="grammar",
        budget_key="grammar",
        retrieval_fn=mock_fn,
        append_fn=_append_rule_entries,
        title="Грамматические ориентиры:",
        kb_attr=None,
        uses_structural_call=False,
        candidate_attr="grammar_candidates",
    )
    test_registry = [test_config]

    with patch("src.prompt_builder.KB_BLOCK_REGISTRY", test_registry):
        result = builder.build(
            text="Тестовый текст без совпадений.",
            domain="blog",
            include_knowledge=True,
            include_few_shot=False,
            knowledge_level=KnowledgeLevel.FULL,
            token_budget=None,
        )

    note_pos = result.find("теме раздела")
    header_pos = result.find("Грамматические ориентиры:")
    assert note_pos != -1, "Квалификатор не найден в промпте"
    assert header_pos != -1, "Заголовок блока не найден в промпте"
    assert note_pos < header_pos, "Квалификатор должен стоять перед заголовком блока"


# ============================================================================
# Новые тесты ТП-1 (исправление _collect_with_budget)
# ============================================================================

def test_collect_with_budget_applies_to_first_entry() -> None:
    """
    Проверяет, что char_budget применяется даже к первой записи.
    Раньше из-за условия `and result` первая запись всегда проходила,
    теперь исправлено.
    """
    huge_entry = {
        "wrong": "x" * 5000,
        "correct": "y" * 5000,
        "rule": "z" * 1000,
    }
    entries = [huge_entry]
    result, dropped = _collect_with_budget(entries, limit=1, char_budget=100)
    assert result == [], "Первая запись не должна быть включена из-за превышения бюджета"
    assert dropped == 1, "Одна запись должна быть отброшена"


# ============================================================================
# НОВЫЕ ТЕСТЫ ДЛЯ ШАГА 10 (исправлены)
# ============================================================================

# ----------------------------------------------------------------------------
# 1. _process_kb_block с моком retrieval_fn
# ----------------------------------------------------------------------------

def test_process_kb_block_grammar() -> None:
    """Проверяет, что _process_kb_block правильно обрабатывает блок grammar."""
    fake_entry = {"wrong": "ошибка", "rule": "правило", "tags": ["grammar"]}
    mock_fn = MagicMock(return_value=([fake_entry], FallbackStage.STRONG, 0))

    config = KBBlockConfig(
        name="grammar",
        budget_key="grammar",
        retrieval_fn=mock_fn,
        append_fn=_append_rule_entries,
        title="Грамматические ориентиры:",
        kb_attr=None,
        uses_structural_call=False,
        candidate_attr="grammar_candidates",
    )

    lines: list[str] = []
    meta: dict = {}
    kb = SimpleNamespace(
        grammar_errors=[],
        stylistic_issues=[],
        logic_issues=[],
        composition_principles=[],
        composition_errors=[],
        local_cohesion=[],
        storytelling_frameworks=[],
        marketing_templates=[],
        rhetoric_frameworks=[],
        editorial_techniques=[],
        stop_words={},
        domain_glossary={},
        nkrj_structure_patterns={},
    )
    budget = BlockBudget(entry_limit=5, char_budget=None, enabled=True)
    limits = LimitsConfig()

    total = _process_kb_block(
        config=config,
        lines=lines,
        meta=meta,
        kb=kb,
        text="Тестовый текст",
        primary_tags=set(),
        expanded_tags=set(),
        budget=budget,
        domain="blog",
        intent=None,
        overlays=[],
        include_few_shot=False,
        total_few_shot_used=0,
        limits=limits,
        few_shot_seed=None,
    )

    assert len(lines) > 0
    assert "Грамматические ориентиры:" in lines[0]
    assert "ошибка" in lines[1]
    assert total == 0
    mock_fn.assert_called_once()


# ----------------------------------------------------------------------------
# 2. Проверка, что _build_knowledge_block не вызывает _process_kb_block для disabled
# ----------------------------------------------------------------------------

def test_process_kb_block_skips_disabled(builder: PromptBuilder) -> None:
    """Блок с enabled=False не должен обрабатываться в _build_knowledge_block."""
    mock_process = MagicMock()
    with patch('src.prompt_builder._process_kb_block', mock_process):
        budget = KnowledgeBudget({
            "grammar": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
        })
        data = {
            "grammar_errors": [{"wrong": "x", "rule": "y", "tags": ["grammar"]}],
            "stylistic_issues": [],
            "logic_issues": [],
            "composition_principles": [],
            "composition_errors": [],
            "local_cohesion": [],
            "storytelling_frameworks": [],
            "marketing_templates": [],
            "rhetoric_frameworks": [],
            "editorial_techniques": [],
            "stop_words": {},
            "domain_glossary": {},
            "nkrj_structure_patterns": {},
        }
        kb = make_mock_kb(data)
        with patch('src.prompt_builder.load_knowledge_base', return_value=kb):
            builder._build_knowledge_block(
                text="Тест",
                primary_tags=set(),
                expanded_tags=set(),
                budget=budget,
                domain="blog",
                intent=None,
                overlays=[],
                include_few_shot=False,
                total_few_shot_used=0,
            )
        mock_process.assert_not_called()


# ----------------------------------------------------------------------------
# 3. Порядок блоков в _build_knowledge_block (ИСПРАВЛЕН)
# ----------------------------------------------------------------------------

def test_build_knowledge_block_order(builder: PromptBuilder) -> None:
    """Проверяем, что блоки выводятся в порядке, заданном в KB_BLOCK_REGISTRY."""
    def mock_process(*args, **kwargs):
        config = kwargs["config"]
        lines = kwargs["lines"]
        lines.append(config.title)
        return kwargs.get('total_few_shot_used', 0)

    with patch('src.prompt_builder._process_kb_block', side_effect=mock_process) as mock_proc:
        data = {
            "grammar_errors": [{"wrong": "g"}],
            "stylistic_issues": [{"wrong": "s"}],
            "logic_issues": [{"wrong": "l"}],
            "composition_principles": [{"name": "c1"}],
            "composition_errors": [{"name": "ce1"}],
            "local_cohesion": [{"name": "coh1"}],
            "storytelling_frameworks": [{"name": "st1"}],
            "marketing_templates": [{"name": "m1"}],
            "rhetoric_frameworks": [{"name": "r1"}],
            "editorial_techniques": [{"name": "e1"}],
            "stop_words": {},
            "domain_glossary": {},
            "nkrj_structure_patterns": {},
        }
        kb = make_mock_kb(data)
        with patch('src.prompt_builder.load_knowledge_base', return_value=kb):
            budget_dict = {
                block.budget_key: BlockBudget(entry_limit=10, char_budget=None, enabled=True)
                for block in KB_BLOCK_REGISTRY
            }
            budget = KnowledgeBudget(budget_dict)
            # Явно включаем все блоки, чтобы проверить порядок
            text, _, _ = builder._build_knowledge_block(
                text="Тест",
                primary_tags=set(),
                expanded_tags=set(),
                budget=budget,
                domain="blog",
                intent=None,
                overlays=[],
                include_few_shot=False,
                total_few_shot_used=0,
                storytelling_enabled=True,
                marketing_enabled=True,
                antiai_enabled=True,
                rhetoric_enabled=True,
                nkrj_enabled=True,
                editorial_enabled=True,
            )

    expected_titles = [block.title for block in KB_BLOCK_REGISTRY]
    titles_in_text = [line.strip() for line in text.splitlines() if line.strip() in expected_titles]
    assert titles_in_text == expected_titles, f"Порядок блоков нарушен: {titles_in_text} != {expected_titles}"
    assert mock_proc.call_count == len(KB_BLOCK_REGISTRY)


# ----------------------------------------------------------------------------
# 4. allow_storytelling=False (ИСПРАВЛЕН)
# ----------------------------------------------------------------------------

def test_allow_storytelling_false(builder: PromptBuilder) -> None:
    """При allow_storytelling=False блок storytelling отсутствует в промпте."""
    domain_config = DomainConfig(
        name="blog",
        system_rules="",
        tone="neutral",
        allow_storytelling=False,
        allow_marketing=True,
    )

    called_blocks = []

    def mock_process(*args, **kwargs):
        config = kwargs["config"]
        lines = kwargs["lines"]
        called_blocks.append(config.name)
        lines.append(config.title)
        return kwargs.get('total_few_shot_used', 0)

    # Мокаем load_domain_config, чтобы он возвращал наш domain_config
    with patch("src.prompt_builder.load_domain_config", return_value=domain_config):
        with patch('src.prompt_builder._process_kb_block', side_effect=mock_process):
            builder.build(
                text="Тестовый текст.",
                domain="blog",
                include_knowledge=True,
                knowledge_level=KnowledgeLevel.FULL,
                include_few_shot=False,
            )

    assert "storytelling" not in called_blocks, "Блок storytelling был вызван, хотя должен быть отключён"
    assert "grammar" in called_blocks
    assert "style" in called_blocks


# ----------------------------------------------------------------------------
# 5. Детерминированный few-shot seed
# ----------------------------------------------------------------------------

def test_few_shot_seed_determinism(builder: PromptBuilder) -> None:
    """Два вызова build с одинаковым seed дают идентичный промпт."""
    text = "Он согласился согласно приказа начальника."
    result1 = builder.build(
        text=text,
        domain="blog",
        include_knowledge=True,
        include_few_shot=True,
        few_shot_seed=42,
        token_budget=5000,
    )
    result2 = builder.build(
        text=text,
        domain="blog",
        include_knowledge=True,
        include_few_shot=True,
        few_shot_seed=42,
        token_budget=5000,
    )
    assert result1 == result2, "Промпты должны быть идентичны при одинаковом seed"


# ----------------------------------------------------------------------------
# 6. _derive_seed стабильность
# ----------------------------------------------------------------------------

def test_derive_seed_stable() -> None:
    """Проверяем, что _derive_seed даёт одинаковый seed для одинакового текста."""
    text_a = "одинаковый текст"
    text_b = "другой текст"
    assert _derive_seed(text_a) == _derive_seed(text_a)
    assert _derive_seed(text_b) == _derive_seed(text_b)
    assert _derive_seed(text_a) != _derive_seed(text_b)


# ----------------------------------------------------------------------------
# 7. DeprecationWarning для build_prompt (исправлен)
# ----------------------------------------------------------------------------

def test_deprecation_warning(builder: PromptBuilder) -> None:
    """Вызов build_prompt должен вызывать DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(DeprecationWarning, match=r"build_prompt\(\) is deprecated, use build\(\)"):
            builder.build_prompt(
                text="Тест",
                domain="blog",
                include_knowledge=False,
            )


# ----------------------------------------------------------------------------
# 8. reload_configs очищает кеш (ИСПРАВЛЕН)
# ----------------------------------------------------------------------------

def test_reload_clears_cache(builder: PromptBuilder) -> None:
    """reload_configs должен очищать кэш доменов и интентов."""
    # Прогреваем кэш
    builder.get_domain_config("blog")
    builder.get_intent_config("storytelling")

    # Мокаем load_domain_config и load_intent_config
    with patch("src.prompt_builder.load_domain_config", wraps=load_domain_config) as mock_load_domain, \
         patch("src.prompt_builder.load_intent_config", wraps=load_intent_config) as mock_load_intent:
        # Сброс кэша
        builder.reload_configs()
        # Повторные вызовы должны вызвать load_* снова
        builder.get_domain_config("blog")
        builder.get_intent_config("storytelling")
        assert mock_load_domain.call_count == 1
        assert mock_load_intent.call_count == 1


# ----------------------------------------------------------------------------
# 9. KnowledgeBudget.disable
# ----------------------------------------------------------------------------

def test_budget_disable() -> None:
    """Проверяем, что disable отключает указанный блок."""
    budgets = {
        "grammar": BlockBudget(entry_limit=5, char_budget=100, enabled=True),
        "style": BlockBudget(entry_limit=5, char_budget=100, enabled=True),
    }
    budget = KnowledgeBudget(budgets)

    budget.disable("grammar")
    assert budget.get("grammar").enabled is False
    assert budget.get("style").enabled is True

    budget.disable("nonexistent")  # молча игнорируется
    assert budget.get("grammar").enabled is False
    assert budget.get("grammar").entry_limit == 5
    assert budget.get("grammar").char_budget == 100


# ============================================================================
# НОВЫЕ ТЕСТЫ ДЛЯ ЗАДАЧ 4-8 (исправлены)
# ============================================================================

def test_domain_tasks_and_constraints_in_prompt(builder: PromptBuilder) -> None:
    """Проверяет, что в промпт добавляются блоки 'Задачи редактора' и 'Ограничения домена'."""
    prompt = builder.build(text="Тест", domain="blog", include_knowledge=False)
    assert "Задачи редактора" in prompt
    assert "Ограничения домена" in prompt


def test_ip_ceiling_in_prompt(builder: PromptBuilder) -> None:
    """Проверяет наличие целевого ИП в промпте."""
    prompt = builder.build(text="Тест", domain="deai", include_knowledge=False)
    assert "Целевой Индекс пластиковости" in prompt
    assert "≤ 1.7" in prompt  # для deai

    prompt_blog = builder.build(text="Тест", domain="blog", include_knowledge=False)
    assert "≤ 2.5" in prompt_blog  # глобальный


def test_conflicting_overlays_resolved_not_raised(builder: PromptBuilder) -> None:
    """
    Проверяет, что конфликтующие оверлеи с явным suppress разрешаются без ошибки.
    Ранее тест ожидал ValueError, но теперь конфликт разрешается через suppress.
    """
    prompt = builder.build(
        text="Тест",
        domain="blog",
        overlays=["finalcheck_full", "finalcheck_light"],
        include_knowledge=False,
    )
    # Ожидаем, что останется только один из них (finalcheck_full побеждает)
    # Проверяем, что finalcheck_full есть, а finalcheck_light отсутствует
    assert "finalcheck_full" in prompt
    assert "finalcheck_light" not in prompt


def test_non_conflicting_overlays_ok(builder: PromptBuilder) -> None:
    """Проверяет, что неконфликтующие оверлеи работают."""
    prompt = builder.build(
        text="Тест",
        domain="blog",
        overlays=["factcheck", "infostyle"],
        include_knowledge=False,
    )
    assert isinstance(prompt, str)
    assert "Overlay-инструкции" in prompt


def test_load_output_format_no_markdown_removed() -> None:
    """Проверяет, что в load_output_format не используется no_markdown_note."""
    result = load_output_format("text_only")
    assert "Markdown" in result
    assert "no_markdown" not in result


# ============================================================================
# НОВЫЕ ТЕСТЫ ДЛЯ edit_level (Этап 5) — ИСПРАВЛЕНЫ
# ============================================================================

def test_edit_level_default_processing(builder: PromptBuilder) -> None:
    """При отсутствии edit_level в JSON значение по умолчанию 'processing'."""
    import tempfile
    import json
    from pathlib import Path
    from src.prompt_builder import load_domain_config

    with tempfile.TemporaryDirectory() as tmp:
        config_dir = Path(tmp) / "config" / "domains"
        config_dir.mkdir(parents=True)
        domain_file = config_dir / "test_domain.json"
        domain_data = {
            "name": "test_domain",
            "system_rules": "",
            "tone": "neutral",
            "allow_storytelling": False,
            "allow_marketing": False,
        }
        domain_file.write_text(json.dumps(domain_data), encoding="utf-8")

        config = load_domain_config("test_domain", base_path=Path(tmp) / "config")
        assert config.edit_level == "processing"


def test_edit_level_valid_value(builder: PromptBuilder) -> None:
    """Валидное значение edit_level загружается без изменений."""
    import tempfile
    import json
    from pathlib import Path
    from src.prompt_builder import load_domain_config

    with tempfile.TemporaryDirectory() as tmp:
        config_dir = Path(tmp) / "config" / "domains"
        config_dir.mkdir(parents=True)
        domain_file = config_dir / "test_domain.json"
        domain_data = {
            "name": "test_domain",
            "system_rules": "",
            "tone": "neutral",
            "edit_level": "adaptive_remake",
        }
        domain_file.write_text(json.dumps(domain_data), encoding="utf-8")

        config = load_domain_config("test_domain", base_path=Path(tmp) / "config")
        assert config.edit_level == "adaptive_remake"


@pytest.mark.parametrize("invalid_value", [None, 123, [], "unsafe_rewrite"])
def test_edit_level_invalid_fallback(
    builder: PromptBuilder,
    caplog: pytest.LogCaptureFixture,
    invalid_value,
) -> None:
    """Невалидные значения приводят к fallback 'processing' и логу предупреждения."""
    import tempfile
    import json
    from pathlib import Path
    from src.prompt_builder import load_domain_config

    with tempfile.TemporaryDirectory() as tmp:
        config_dir = Path(tmp) / "config" / "domains"
        config_dir.mkdir(parents=True)
        domain_file = config_dir / "test_domain.json"
        domain_data = {
            "name": "test_domain",
            "system_rules": "",
            "tone": "neutral",
            "edit_level": invalid_value,
        }
        domain_file.write_text(json.dumps(domain_data), encoding="utf-8")

        with caplog.at_level("WARNING", logger="src.prompt_builder"):
            config = load_domain_config("test_domain", base_path=Path(tmp) / "config")
            assert config.edit_level == "processing"
            assert "недопустимый edit_level" in caplog.text
            assert "используется 'processing'" in caplog.text


# ИСПРАВЛЕННАЯ параметризация: используем ключевые фразы, которые есть в новом тексте
@pytest.mark.parametrize("level,expected_phrase", [
    ("light", "точечная"),                      # уникальная фраза для light
    ("processing", "Композицию и порядок абзацев не менять"),
    ("remake", "Разрешена перестройка композиции"),
    ("adaptive_remake", "адаптивная переделка"),  # уникальная фраза для adaptive_remake
])
def test_build_edit_level_block_contains_key_phrases(builder: PromptBuilder, level: str, expected_phrase: str) -> None:
    """Проверяет, что _build_edit_level_block возвращает правильные фразы для каждого уровня."""
    domain_config = DomainConfig(
        name="test",
        system_rules="",
        tone="neutral",
        edit_level=level,
    )
    block = builder._build_edit_level_block(domain_config)
    assert expected_phrase in block, f"Фраза '{expected_phrase}' не найдена для уровня {level}"


def test_build_edit_level_block_integration(builder: PromptBuilder) -> None:
    """Интеграционный тест: для домена genre с overlay casestudy, проверяем наличие adaptive_remake и условной логики."""
    prompt = builder.build(
        text="Мы делаем сайты уже пять лет. У нас хорошая команда.",
        domain="genre",
        overlays=["casestudy"],
        include_knowledge=False,
    )
    assert "Уровень правки: адаптивная переделка" in prompt
    assert "Если текст уже соответствует" in prompt   # проверяем условную конструкцию
    assert "Если текст не соответствует" in prompt


def test_build_edit_level_for_domain_without_edit_level(builder: PromptBuilder) -> None:
    """Для домена без edit_level (например, blog) выводится 'обработка'."""
    prompt = builder.build(
        text="Тестовый текст",
        domain="blog",
        include_knowledge=False,
    )
    assert "Уровень правки: обработка" in prompt
    assert "Композицию и порядок абзацев не менять" in prompt
