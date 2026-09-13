# tests/test_prompt_builder.py
from __future__ import annotations

import re
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config_types import (
    AssemblyTrace,
    BlockBudget,
    DomainConfig,
    KnowledgeBudget,
    KnowledgeLevel,
    LimitsConfig,
)
from src.knowledge_retrieval import FallbackStage, _collect_with_budget
from src.prompt_builder import (
    KB_BLOCK_REGISTRY,
    KBBlockConfig,
    KnowledgeBlockRequest,
    PromptBuilder,
    load_domain_config,
    load_intent_config,
    load_output_format,
)
from src.prompt_builder.kb_rendering import (
    ProcessContext,
    _append_rule_entries,
    _derive_seed,
    _get_confidence_note,
    _process_kb_block,
)
from src.reason_codes import ReasonCode

# ============================================================================
# Вспомогательная функция нормализации nonce для тестов
# ============================================================================


def _normalize_user_text_markers(text: str) -> str:
    """
    Заменяет случайный nonce в USER_TEXT-маркерах на фиксированную строку.

    Маркеры имеют вид <<<USER_TEXT_<hex8>_START>>> и <<<USER_TEXT_<hex8>_END>>>,
    где <hex8> генерируется secrets.token_hex(4) при каждом вызове build().
    Эта функция делает их детерминированными для сравнения в тестах.
    """
    return re.sub(
        r"<<<USER_TEXT_[0-9a-f]+_(START|END)>>>",
        r"<<<USER_TEXT_NONCE_\1>>>",
        text,
    )


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
    for key, value in data.items():
        setattr(kb, key, value)

    def get_side_effect(key, default=None):
        return data.get(key, default)

    kb.get.side_effect = get_side_effect
    return kb


# ============================================================================
# Старые тесты
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


def test_knowledge_level_none_disables_knowledge_content(
    builder: PromptBuilder,
) -> None:
    result = builder.build(
        text="Текст с knowledge_level none.",
        domain="blog",
        include_knowledge=True,
        knowledge_level=KnowledgeLevel.NONE,
    )
    assert "Исходный текст:" in result
    assert "База знаний:" not in result


def test_knowledge_level_core_or_standard_does_not_crash(
    builder: PromptBuilder,
) -> None:
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


def test_full_level_without_optional_configs_does_not_crash(
    builder: PromptBuilder,
) -> None:
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
def test_supported_domain_intent_combinations(
    builder: PromptBuilder, domain: str, intent: str
) -> None:
    result = builder.build(
        text="Проверка допустимой комбинации домена и intent.",
        domain=domain,
        intent=intent,
        include_knowledge=False,
    )
    assert isinstance(result, str)
    assert "Домен:" in result


# ---------------------------------------------------------------------------
# Few-shot тесты
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


def test_include_few_shot_without_knowledge_does_nothing(
    builder: PromptBuilder,
) -> None:
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
    assert _normalize_user_text_markers(result) == _normalize_user_text_markers(result2)


# ---------------------------------------------------------------------------
# Тесты квалификатора уверенности
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

    with patch("src.prompt_builder.builder.KB_BLOCK_REGISTRY", test_registry):
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
    """При stage=STRONG квалификатор не должен появляться."""
    data = {
        "grammar_errors": [
            {
                "wrong": "Тестовый текст",
                "rule": "правило грамматики",
                "tags": ["grammar"],
            }
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

    with patch("src.prompt_builder.builder.load_knowledge_base", return_value=kb):
        result = builder.build(
            text="Тестовый текст",
            domain="blog",
            include_knowledge=True,
            include_few_shot=False,
            knowledge_level=KnowledgeLevel.FULL,
            token_budget=None,
        )
        assert "Грамматические ориентиры:" in result
        assert "теме раздела" not in result
        assert "смысловому совпадению" not in result


def test_confidence_note_not_inserted_when_no_knowledge(builder: PromptBuilder) -> None:
    """Если блоки знаний пусты, квалификатор не добавляется."""
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

    with patch("src.prompt_builder.builder.load_knowledge_base", return_value=kb):
        result = builder.build(
            text="Тестовый текст.",
            domain="blog",
            include_knowledge=True,
            include_few_shot=False,
            knowledge_level=KnowledgeLevel.FULL,
            token_budget=None,
        )
        assert "теме раздела" not in result
        assert "смысловому совпадению" not in result
        assert "Грамматические ориентиры:" not in result


def test_confidence_note_appears_only_once_per_block(builder: PromptBuilder) -> None:
    """Квалификатор добавляется ровно один раз для блока."""
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

    with patch("src.prompt_builder.builder.KB_BLOCK_REGISTRY", test_registry):
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
    """Проверяем, что квалификатор идёт перед заголовком "Грамматические ориентиры:"."""
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

    with patch("src.prompt_builder.builder.KB_BLOCK_REGISTRY", test_registry):
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
# Тесты _collect_with_budget
# ============================================================================


def test_collect_with_budget_applies_to_first_entry() -> None:
    """Проверяет, что char_budget применяется даже к первой записи."""
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
# Тесты для _process_kb_block
# ============================================================================


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

    ctx = ProcessContext(
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
        semantic_rerank=False,
    )

    total = _process_kb_block(config=config, ctx=ctx)

    assert len(lines) > 0
    assert "Грамматические ориентиры:" in lines[0]
    assert "ошибка" in lines[1]
    assert total == 0
    mock_fn.assert_called_once()


# ----------------------------------------------------------------------------
# Проверка, что _build_knowledge_block не вызывает _process_kb_block для disabled
# ----------------------------------------------------------------------------


def test_process_kb_block_skips_disabled(builder: PromptBuilder) -> None:
    """Блок с enabled=False не должен обрабатываться в _build_knowledge_block."""
    mock_process = MagicMock()
    with patch("src.prompt_builder.builder._process_kb_block", mock_process):
        budget = KnowledgeBudget(
            {
                "grammar": BlockBudget(entry_limit=5, char_budget=None, enabled=False),
            }
        )
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
        with patch("src.prompt_builder.builder.load_knowledge_base", return_value=kb):
            req = KnowledgeBlockRequest(
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
            builder._build_knowledge_block(req)
        mock_process.assert_not_called()


# ----------------------------------------------------------------------------
# Порядок блоков в _build_knowledge_block
# ----------------------------------------------------------------------------


def test_build_knowledge_block_order(builder: PromptBuilder) -> None:
    """Проверяем, что блоки выводятся в порядке, заданном в KB_BLOCK_REGISTRY."""

    def mock_process(config, ctx):
        lines = ctx.lines
        lines.append(config.title)
        return ctx.total_few_shot_used

    with patch(
        "src.prompt_builder.builder._process_kb_block", side_effect=mock_process
    ) as mock_proc:
        data = {
            "grammar_errors": [{"wrong": "g"}],
            "stylistic_issues": [{"wrong": "s"}],
            "logic_issues": [{"wrong": "l"}],
            "composition_principles": [{"name": "c1"}],
            "composition_errors": [{"name": "ce1"}],
            "local_cohesion": [{"name": "coh1"}],
            "storytelling_frameworks": [{"name": "st1"}],
            "marketing_templates": [{"name": "m1"}],
            "case_study_templates": [{"name": "cs1"}],
            "rhetoric_frameworks": [{"name": "r1"}],
            "editorial_techniques": [{"name": "e1"}],
            "evaluation_techniques": {"category": "evaluations"},
            "stop_words": {},
            "domain_glossary": {},
            "nkrj_structure_patterns": {},
        }
        kb = make_mock_kb(data)
        with patch("src.prompt_builder.builder.load_knowledge_base", return_value=kb):
            budget_dict = {
                block.budget_key: BlockBudget(entry_limit=10, char_budget=None, enabled=True)
                for block in KB_BLOCK_REGISTRY
            }
            budget = KnowledgeBudget(budget_dict)
            req = KnowledgeBlockRequest(
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
            text, _, _ = builder._build_knowledge_block(req)

    expected_titles = [block.title for block in KB_BLOCK_REGISTRY if block.title]
    titles_in_text = [line.strip() for line in text.splitlines() if line.strip() in expected_titles]
    assert (
        titles_in_text == expected_titles
    ), f"Порядок блоков нарушен: {titles_in_text} != {expected_titles}"
    assert mock_proc.call_count == len(KB_BLOCK_REGISTRY) - 1


# ----------------------------------------------------------------------------
# allow_storytelling=False
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

    def mock_process(config, ctx):
        called_blocks.append(config.name)
        ctx.lines.append(config.title)
        return ctx.total_few_shot_used

    with (
        patch("src.prompt_builder.builder.load_domain_config", return_value=domain_config),
        patch("src.prompt_builder.builder._process_kb_block", side_effect=mock_process),
    ):
        builder.build(
            text="Тестовый текст.",
            domain="blog",
            include_knowledge=True,
            knowledge_level=KnowledgeLevel.FULL,
            include_few_shot=False,
        )

    assert (
        "storytelling" not in called_blocks
    ), "Блок storytelling был вызван, хотя должен быть отключён"
    assert "grammar" in called_blocks
    assert "style" in called_blocks


# ----------------------------------------------------------------------------
# Детерминированный few-shot seed
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
    assert _normalize_user_text_markers(result1) == _normalize_user_text_markers(
        result2
    ), "Промпты должны быть идентичны при одинаковом seed"


# ----------------------------------------------------------------------------
# _derive_seed стабильность
# ----------------------------------------------------------------------------


def test_derive_seed_stable() -> None:
    """Проверяем, что _derive_seed даёт одинаковый seed для одинакового текста."""
    text_a = "одинаковый текст"
    text_b = "другой текст"
    assert _derive_seed(text_a) == _derive_seed(text_a)
    assert _derive_seed(text_b) == _derive_seed(text_b)
    assert _derive_seed(text_a) != _derive_seed(text_b)


# ----------------------------------------------------------------------------
# DeprecationWarning для build_prompt
# ----------------------------------------------------------------------------


def test_deprecation_warning(builder: PromptBuilder) -> None:
    """Вызов build_prompt должен вызывать DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(
            DeprecationWarning, match=r"build_prompt\(\) is deprecated, use build\(\)"
        ):
            builder.build_prompt(
                text="Тест",
                domain="blog",
                include_knowledge=False,
            )


# ----------------------------------------------------------------------------
# reload_configs очищает кеш
# ----------------------------------------------------------------------------


def test_reload_clears_cache(builder: PromptBuilder) -> None:
    """reload_configs должен очищать кэш доменов и интентов."""
    builder.get_domain_config("blog")
    builder.get_intent_config("storytelling")

    with (
        patch(
            "src.prompt_builder.builder.load_domain_config", wraps=load_domain_config
        ) as mock_load_domain,
        patch(
            "src.prompt_builder.builder.load_intent_config", wraps=load_intent_config
        ) as mock_load_intent,
    ):
        builder.reload_configs()
        builder.get_domain_config("blog")
        builder.get_intent_config("storytelling")
        assert mock_load_domain.call_count == 1
        assert mock_load_intent.call_count == 1


# ----------------------------------------------------------------------------
# KnowledgeBudget.disable
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

    budget.disable("nonexistent")
    assert budget.get("grammar").enabled is False
    assert budget.get("grammar").entry_limit == 5
    assert budget.get("grammar").char_budget == 100


# ============================================================================
# Тесты для задач 4-8
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
    assert "≤ 1.7" in prompt

    prompt_blog = builder.build(text="Тест", domain="blog", include_knowledge=False)
    assert "≤ 2.5" in prompt_blog


def test_conflicting_overlays_resolved_not_raised(builder: PromptBuilder) -> None:
    """Проверяет, что конфликтующие оверлеи с явным suppress разрешаются без ошибки."""
    prompt = builder.build(
        text="Тест",
        domain="blog",
        overlays=["finalcheck_full", "finalcheck_light"],
        include_knowledge=False,
    )
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
# Тесты для edit_level
# ============================================================================


def test_edit_level_default_processing(builder: PromptBuilder) -> None:
    """При отсутствии edit_level в JSON значение по умолчанию 'processing'."""
    import json
    import tempfile
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
    import json
    import tempfile
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
    import json
    import tempfile
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


@pytest.mark.parametrize(
    "level,expected_phrase",
    [
        ("light", "точечная"),
        ("processing", "Композицию и порядок абзацев не менять"),
        ("remake", "Разрешена перестройка композиции"),
        ("adaptive_remake", "адаптивная переделка"),
    ],
)
def test_build_edit_level_block_contains_key_phrases(
    builder: PromptBuilder, level: str, expected_phrase: str
) -> None:
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
    """Интеграционный тест: для домена genre с overlay casestudy."""
    prompt = builder.build(
        text="Мы делаем сайты уже пять лет. У нас хорошая команда.",
        domain="genre",
        overlays=["casestudy"],
        include_knowledge=False,
    )
    assert "Уровень правки: адаптивная переделка" in prompt
    assert "Если текст уже соответствует" in prompt
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


# ============================================================================
# Тесты для case_study
# ============================================================================


def test_prompt_includes_case_study_knowledge(builder: PromptBuilder) -> None:
    """Проверяет, что при overlay casestudy в промпт добавляется жанровый блок."""
    prompt = builder.build(
        text="Компания столкнулась с проблемой роста затрат на логистику.",
        domain="genre",
        overlays=["casestudy"],
        include_knowledge=True,
        include_few_shot=False,
        knowledge_level=KnowledgeLevel.FULL,
        include_retrieval_meta=False,
    )
    expected_phrase = "Обязательные элементы структуры кейса"
    assert (
        expected_phrase in prompt
    ), f"В промпте отсутствует фраза '{expected_phrase}'. Промпт (первые 1000 символов):\n{prompt[:1000]}"
    assert "контекст, проблема, решение, результат, вывод" in prompt or "контекст" in prompt
    assert "case_study_composition" not in prompt
    assert "genre_knowledge" not in prompt
    count = prompt.count(expected_phrase)
    assert count == 1, f"Фраза дублируется ({count} раз)"


def test_retrieval_meta_selects_case_study_template(builder: PromptBuilder) -> None:
    """Диагностика: при overlay casestudy retrieval выбирает шаблон по стабильному ID."""
    _, meta = builder.build(
        text="Компания столкнулась с проблемой роста затрат на логистику.",
        domain="genre",
        intent="marketingpush",
        overlays=["base", "casestudy"],
        include_knowledge=True,
        include_few_shot=False,
        knowledge_level=KnowledgeLevel.FULL,
        include_retrieval_meta=True,
    )
    block_meta = meta.get("casestudy")
    assert block_meta is not None, f"Блок casestudy отсутствует в retrieval_meta: {sorted(meta)}"
    assert (
        "case_study_composition" in block_meta["entry_ids"]
    ), f"Шаблон case_study_composition не выбран: {block_meta}"


def test_prompt_without_casestudy_does_not_include_genre_block(
    builder: PromptBuilder,
) -> None:
    """Проверяет, что без оверлея casestudy жанровый блок не появляется."""
    prompt = builder.build(
        text="Обычный маркетинговый текст.",
        domain="marketing",
        overlays=["coldemail"],
        include_knowledge=True,
        include_few_shot=False,
        knowledge_level=KnowledgeLevel.FULL,
    )
    assert "Базовая композиция бизнес-кейса" not in prompt
    assert "Проблема и исходная точка" not in prompt


def test_existing_marketing_blocks_remain(builder: PromptBuilder) -> None:
    """Регрессионный тест: старые маркетинговые шаблоны не исчезли после добавления case_study."""
    prompt = builder.build(
        text="Текст для email-рассылки о новом продукте.",
        domain="marketing",
        overlays=["coldemail"],
        include_knowledge=True,
        include_few_shot=False,
        knowledge_level=KnowledgeLevel.FULL,
    )
    assert (
        "Продуктовое письмо" in prompt or "Лендинг" in prompt
    ), "Старые маркетинговые шаблоны отсутствуют в промпте"
    assert "Базовая композиция бизнес-кейса" not in prompt


# ============================================================================
# Тесты load_full_kb
# ============================================================================


def test_load_full_kb(builder: PromptBuilder) -> None:
    """Проверяет, что load_full_kb загружает KB и сохраняет в _loaded_kb."""
    assert builder._loaded_kb is None

    kb = builder.load_full_kb()
    assert kb is not None
    assert hasattr(kb, "grammar_errors")
    assert builder._loaded_kb is kb

    kb2 = builder.load_full_kb()
    assert kb2 is kb


def test_load_full_kb_uses_load_all(builder: PromptBuilder) -> None:
    """Проверяет, что load_full_kb вызывает load_knowledge_base с load_all=True."""
    with patch("src.prompt_builder.builder.load_knowledge_base") as mock_load:
        mock_load.return_value = MagicMock()
        builder.load_full_kb()
        mock_load.assert_called_once_with(
            builder.kb_path,
            active_tags=None,
            intent=None,
            load_all=True,
        )


def test_reload_configs_clears_loaded_kb(builder: PromptBuilder) -> None:
    """Проверяет, что reload_configs сбрасывает _loaded_kb."""
    builder.load_full_kb()
    assert builder._loaded_kb is not None

    builder.reload_configs()
    assert builder._loaded_kb is None


def test_invalidate_caches_clears_loaded_kb(builder: PromptBuilder) -> None:
    """Проверяет, что _invalidate_caches сбрасывает _loaded_kb."""
    builder.load_full_kb()
    assert builder._loaded_kb is not None

    builder._invalidate_caches()
    assert builder._loaded_kb is None


# ============================================================================
# Тесты для разбиения _build_knowledge_block
# ============================================================================


def test_process_stop_words_block(builder: PromptBuilder) -> None:
    """Проверяет _process_stop_words_block."""
    from src.config_types import BlockBudget, KnowledgeBudget

    kb = make_mock_kb({"stop_words": {"category1": ["word1", "word2"], "category2": ["word3"]}})
    req = KnowledgeBlockRequest(
        text="test",
        primary_tags=set(),
        expanded_tags=set(),
        budget=KnowledgeBudget(
            {"stop_words": BlockBudget(entry_limit=10, char_budget=None, enabled=True)}
        ),
        domain="blog",
        intent=None,
        overlays=[],
        include_few_shot=False,
        total_few_shot_used=0,
        limits=LimitsConfig(stop_words_category=2, stop_words_items=2),
    )
    lines: list[str] = []
    builder._process_stop_words_block(kb, req, lines, None, req.limits)
    assert len(lines) == 3
    assert "Стоп-слова" in lines[0]
    assert "category1: word1, word2" in lines[1]


def test_process_evaluation_block(builder: PromptBuilder) -> None:
    """Проверяет _process_evaluation_block."""
    kb = make_mock_kb({"evaluation_techniques": {"technique1": "description"}})
    req = KnowledgeBlockRequest(
        text="test",
        primary_tags=set(),
        expanded_tags=set(),
        budget=KnowledgeBudget({}),
        domain="blog",
        intent=None,
        overlays=[],
        include_few_shot=False,
        total_few_shot_used=0,
    )
    lines: list[str] = []
    builder._process_evaluation_block(kb, req, lines, None)
    assert any("Техники работы с оценками" in line for line in lines)


def test_process_glossary_block(builder: PromptBuilder) -> None:
    """Проверяет _process_glossary_block."""
    from src.config_types import BlockBudget, KnowledgeBudget

    kb = make_mock_kb({"domain_glossary": {"term1": "definition1", "term2": "definition2"}})
    req = KnowledgeBlockRequest(
        text="test",
        primary_tags=set(),
        expanded_tags=set(),
        budget=KnowledgeBudget(
            {"glossary": BlockBudget(entry_limit=2, char_budget=None, enabled=True)}
        ),
        domain="blog",
        intent=None,
        overlays=[],
        include_few_shot=False,
        total_few_shot_used=0,
    )
    lines: list[str] = []
    builder._process_glossary_block(kb, req, lines, None)
    assert len(lines) >= 2


def test_process_nkrj_block(builder: PromptBuilder) -> None:
    """Проверяет _process_nkrj_block."""
    from src.config_types import BlockBudget, KnowledgeBudget

    kb = make_mock_kb(
        {
            "nkrj_structure_patterns": {
                "pattern1": "description1",
                "pattern2": "description2",
            }
        }
    )
    req = KnowledgeBlockRequest(
        text="test",
        primary_tags=set(),
        expanded_tags=set(),
        budget=KnowledgeBudget(
            {"nkrj": BlockBudget(entry_limit=5, char_budget=None, enabled=True)}
        ),
        domain="blog",
        intent=None,
        overlays=[],
        include_few_shot=False,
        total_few_shot_used=0,
        nkrj_enabled=True,
    )
    lines: list[str] = []
    builder._process_nkrj_block(kb, req, lines, None)
    assert len(lines) > 0, "Блок nkrj не добавил строк"
    data = kb.nkrj_structure_patterns
    found = False
    for line in lines:
        for key, value in data.items():
            if key in line or value in line:
                found = True
                break
        if found:
            break
    assert found, "В lines не найдено содержимое nkrj-паттернов"


def test_add_trace_diagnostic(builder: PromptBuilder) -> None:
    """Проверяет _add_trace_diagnostic."""
    trace = AssemblyTrace()
    builder._add_trace_diagnostic(
        trace,
        "test_block",
        eligible=True,
        included=True,
        reason_codes=[ReasonCode.BLOCK_INCLUDED],
        empty=False,
        char_count=10,
        entries_count=2,
    )
    assert len(trace.blocks) == 1
    diag = trace.blocks[0]
    assert diag.name == "test_block"
    assert diag.eligible is True
    assert diag.included is True
    assert ReasonCode.BLOCK_INCLUDED in diag.reason_codes
    assert diag.char_count == 10
    assert diag.entries_count == 2


# ---------------------------------------------------------------------------
# Тесты для _process_registry_block
# ---------------------------------------------------------------------------


def test_process_registry_block_includes_block_when_eligible():
    """Проверяет, что _process_registry_block включает блок при eligibility."""
    builder = PromptBuilder()
    lines = []
    meta = {}
    req = KnowledgeBlockRequest(
        text="test",
        primary_tags=set(),
        expanded_tags=set(),
        budget=KnowledgeBudget(
            {"grammar": BlockBudget(entry_limit=5, char_budget=None, enabled=True)}
        ),
        domain="blog",
        intent=None,
        overlays=[],
        include_few_shot=False,
        total_few_shot_used=0,
        limits=LimitsConfig(),
        storytelling_enabled=True,
        marketing_enabled=False,
        antiai_enabled=False,
        rhetoric_enabled=False,
        nkrj_enabled=False,
        editorial_enabled=False,
        return_trace=True,
        semantic_rerank=False,
    )
    kb = make_mock_kb({"grammar_errors": [{"wrong": "x", "rule": "y", "tags": ["grammar"]}]})
    block_cfg = KBBlockConfig(
        name="grammar",
        budget_key="grammar",
        retrieval_fn=lambda *args, **kwargs: (
            [{"wrong": "x"}],
            FallbackStage.STRONG,
            0,
        ),
        append_fn=_append_rule_entries,
        title="Грамматические ориентиры:",
        kb_attr=None,
        uses_structural_call=False,
        candidate_attr="grammar_candidates",
    )
    trace = AssemblyTrace()
    current_total = builder._process_registry_block(
        block_cfg,
        kb,
        req,
        lines,
        meta,
        0,
        trace,
        LimitsConfig(),
    )

    assert current_total == 0
    assert len(lines) > 0
    assert "Грамматические ориентиры:" in lines[0]
    assert trace.blocks[0].included is True


def test_process_registry_block_skips_block_when_budget_disabled():
    """Проверяет, что блок пропускается при отключённом бюджете."""
    builder = PromptBuilder()
    lines = []
    meta = {}
    req = KnowledgeBlockRequest(
        text="test",
        primary_tags=set(),
        expanded_tags=set(),
        budget=KnowledgeBudget(
            {"grammar": BlockBudget(entry_limit=5, char_budget=None, enabled=False)}
        ),
        domain="blog",
        intent=None,
        overlays=[],
        include_few_shot=False,
        total_few_shot_used=0,
        limits=LimitsConfig(),
        storytelling_enabled=False,
        marketing_enabled=False,
        antiai_enabled=False,
        rhetoric_enabled=False,
        nkrj_enabled=False,
        editorial_enabled=False,
        return_trace=True,
        semantic_rerank=False,
    )
    kb = make_mock_kb({"grammar_errors": [{"wrong": "x"}]})
    block_cfg = KBBlockConfig(
        name="grammar",
        budget_key="grammar",
        retrieval_fn=None,
        append_fn=None,
        title="",
        kb_attr=None,
        uses_structural_call=False,
        candidate_attr=None,
    )
    trace = AssemblyTrace()
    builder._process_registry_block(block_cfg, kb, req, lines, meta, 0, trace, LimitsConfig())
    assert len(lines) == 0
    assert trace.blocks[0].eligible is False
    assert ReasonCode.BLOCK_INELIGIBLE_BUDGET_DISABLED in trace.blocks[0].reason_codes
