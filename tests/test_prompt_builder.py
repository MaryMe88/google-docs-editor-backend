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
    load_output_format,  # добавлен импорт для задачи 8
)
from src.knowledge_retrieval import FallbackStage, _collect_with_budget


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


def test_invalid_domain_raises_value_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Unsupported domain"):
        builder.build(text="Текст.", domain="science")


def test_invalid_intent_raises_value_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Unsupported intent"):
        builder.build(text="Текст.", domain="blog", intent="unknown_intent")


def test_invalid_overlay_raises_value_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Unsupported overlays"):
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
    mock_fn.assert_called_once()  # проверяем, что наш мок был вызван


def test_confidence_note_not_inserted_for_strong_stage(builder: PromptBuilder) -> None:
    """
    При stage=STRONG квалификатор не должен появляться.
    Используем запись с точным совпадением по wrong (без correct, чтобы это было правило).
    """
    kb = SimpleNamespace(
        grammar_errors=[
            {"wrong": "Тестовый текст", "rule": "правило грамматики", "tags": ["grammar"]}
        ],
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

    with patch.object(builder, '_ensure_knowledge_base', return_value=kb):
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

    with patch.object(builder, '_ensure_knowledge_base', return_value=kb):
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
    # Запись без correct, чтобы она была правилом, а не парой (иначе попадёт в few-shot)
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
        kb = SimpleNamespace(
            grammar_errors=[{"wrong": "x", "rule": "y", "tags": ["grammar"]}],
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
        with patch.object(builder, '_ensure_knowledge_base', return_value=kb):
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
# 3. Порядок блоков в _build_knowledge_block
# ----------------------------------------------------------------------------

def test_build_knowledge_block_order(builder: PromptBuilder) -> None:
    """Проверяем, что блоки выводятся в порядке, заданном в KB_BLOCK_REGISTRY."""
    # Исправлено: используем kwargs, так как _process_kb_block вызывается с именованными аргументами
    def mock_process(*args, **kwargs):
        config = kwargs["config"]
        lines = kwargs["lines"]
        lines.append(config.title)
        return kwargs.get('total_few_shot_used', 0)

    with patch('src.prompt_builder._process_kb_block', side_effect=mock_process) as mock_proc:
        kb = SimpleNamespace(
            grammar_errors=[{"wrong": "g"}],
            stylistic_issues=[{"wrong": "s"}],
            logic_issues=[{"wrong": "l"}],
            composition_principles=[{"name": "c1"}],
            composition_errors=[{"name": "ce1"}],
            local_cohesion=[{"name": "coh1"}],
            storytelling_frameworks=[{"name": "st1"}],
            marketing_templates=[{"name": "m1"}],
            rhetoric_frameworks=[{"name": "r1"}],
            editorial_techniques=[{"name": "e1"}],
            stop_words={},
            domain_glossary={},
            nkrj_structure_patterns={},
        )
        with patch.object(builder, '_ensure_knowledge_base', return_value=kb):
            budget_dict = {
                block.budget_key: BlockBudget(entry_limit=10, char_budget=None, enabled=True)
                for block in KB_BLOCK_REGISTRY
            }
            budget = KnowledgeBudget(budget_dict)
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
            )

    expected_titles = [block.title for block in KB_BLOCK_REGISTRY]
    titles_in_text = [line.strip() for line in text.splitlines() if line.strip() in expected_titles]
    assert titles_in_text == expected_titles, f"Порядок блоков нарушен: {titles_in_text} != {expected_titles}"
    assert mock_proc.call_count == len(KB_BLOCK_REGISTRY)


# ----------------------------------------------------------------------------
# 4. allow_storytelling=False
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

    # Исправлено: используем kwargs
    def mock_process(*args, **kwargs):
        config = kwargs["config"]
        lines = kwargs["lines"]
        called_blocks.append(config.name)
        lines.append(config.title)
        return kwargs.get('total_few_shot_used', 0)

    with patch("src.prompt_builder._cached_load_domain_config", return_value=domain_config):
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
    # Сбрасываем фильтр "показывать только раз", чтобы предупреждение точно было видно
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        # Экранируем скобки в match
        with pytest.warns(DeprecationWarning, match=r"build_prompt\(\) is deprecated, use build\(\)"):
            builder.build_prompt(
                text="Тест",
                domain="blog",
                include_knowledge=False,
            )


# ----------------------------------------------------------------------------
# 8. reload_configs очищает кеш
# ----------------------------------------------------------------------------

def test_reload_clears_cache(builder: PromptBuilder) -> None:
    """reload_configs должен очищать кеш _cached_load_domain_config и _cached_load_intent_config."""
    with patch("src.prompt_builder._cached_load_domain_config") as mock_domain_cache, \
         patch("src.prompt_builder._cached_load_intent_config") as mock_intent_cache:
        mock_domain_cache.cache_clear = MagicMock()
        mock_intent_cache.cache_clear = MagicMock()

        builder.reload_configs()

        mock_domain_cache.cache_clear.assert_called_once()
        mock_intent_cache.cache_clear.assert_called_once()


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
    assert "≤ 1.0" in prompt  # для deai

    prompt_blog = builder.build(text="Тест", domain="blog", include_knowledge=False)
    assert "≤ 2.5" in prompt_blog  # глобальный


def test_conflicting_overlays_raise_error(builder: PromptBuilder) -> None:
    """Проверяет, что конфликтующие оверлеи вызывают ValueError."""
    with pytest.raises(ValueError, match="Overlays conflict"):
        builder.build(
            text="Тест",
            domain="blog",
            overlays=["finalcheck_full", "finalcheck_light"],
            include_knowledge=False,
        )


def test_non_conflicting_overlays_ok(builder: PromptBuilder) -> None:
    """Проверяет, что неконфликтующие оверлеи работают."""
    # Используем существующие оверлеи: factcheck и infostyle
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
    assert "no_markdown" not in result  # ключ не должен фигурировать