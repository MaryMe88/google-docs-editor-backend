import pytest
import json
from pathlib import Path
from typing import List, Dict, Any

from src.knowledge_retrieval import (
    select_grammar_rules,
    select_style_issues,
    select_logic_issues,
    select_structural_by_tags_or_all,
    FallbackStage,
)


def _load_golden_tests() -> List[Dict[str, Any]]:
    """Загружает golden_set.json из папок tests/golden/, tests/ или корня проекта."""
    tests_dir = Path(__file__).parent
    # Возможные пути
    possible_paths = [
        tests_dir / "golden" / "golden_set.json",
        tests_dir / "golden_set.json",
        tests_dir.parent / "golden_set.json",
    ]
    for golden_path in possible_paths:
        if golden_path.exists():
            with open(golden_path, encoding="utf-8") as f:
                data = json.load(f)
            return data["tests"]
    raise FileNotFoundError(
        "golden_set.json not found. Expected locations: " + ", ".join(str(p) for p in possible_paths)
    )


# Загружаем тесты один раз при импорте модуля
GOLDEN_TESTS = _load_golden_tests()


def find_entry_in_results(entries: List[Dict[str, Any]], expected: Dict[str, Any]) -> bool:
    """Проверяет, содержится ли ожидаемая запись среди результатов retrieval."""
    for entry in entries:
        if "expected_wrong" in expected:
            entry_wrong = entry.get("wrong", "")
            if isinstance(entry_wrong, str) and entry_wrong.strip() == expected["expected_wrong"]:
                return True
        if "expected_name" in expected:
            entry_name = entry.get("name", "")
            if isinstance(entry_name, str) and entry_name.strip() == expected["expected_name"]:
                return True
        if "expected_id" in expected:
            if entry.get("id") == expected["expected_id"]:
                return True
    return False


def get_test_id(test_case: Dict[str, Any]) -> str:
    """Возвращает идентификатор для теста."""
    return test_case.get("expected_wrong", test_case.get("expected_name", test_case.get("expected_id", "unknown")))[:40]


@pytest.mark.parametrize("test_case", GOLDEN_TESTS, ids=get_test_id)
def test_golden_retrieval(knowledge_base, test_case):
    """Проверяет, что для каждого эталонного текста retrieval находит ожидаемую запись."""
    text = test_case["text"]
    category = test_case["category"]
    source_file = test_case.get("source_file", "")
    expected_stage = FallbackStage(test_case.get("expected_stage", "strong"))

    # Выбор функции retrieval в зависимости от категории
    if category == "grammar":
        entries, stage, _ = select_grammar_rules(
            kb=knowledge_base,
            text=text,
            tags=["grammar"],
            limit=20,
            return_meta=True,
        )
    elif category == "style":
        entries, stage, _ = select_style_issues(
            kb=knowledge_base,
            text=text,
            tags=["style"],
            limit=20,
            return_meta=True,
        )
    elif category == "logic":
        entries, stage, _ = select_logic_issues(
            kb=knowledge_base,
            text=text,
            tags=["logic"],
            limit=20,
            return_meta=True,
        )
    elif category == "composition":
        # Определяем список записей по source_file
        if "composition_errors" in source_file:
            entries_list = knowledge_base.composition_errors
        elif "composition_principles" in source_file:
            entries_list = knowledge_base.composition_principles
        else:
            entries_list = knowledge_base.composition_errors
        entries, stage, _ = select_structural_by_tags_or_all(
            entries=entries_list,
            tags=["composition"],
            limit=20,
            return_meta=True,
        )
    else:
        pytest.fail(f"Unsupported category: {category}")

    # Порядок стадий (от высшей к низшей)
    stage_order = {
        FallbackStage.STRONG: 5,
        FallbackStage.TEXT_ONLY: 4,
        FallbackStage.TAG_ONLY: 3,
        FallbackStage.NEUTRAL: 2,
        FallbackStage.EMPTY: 1,
    }
    assert stage_order[stage] >= stage_order[expected_stage], \
        f"Stage {stage.value} < {expected_stage.value} for text: {text[:60]}..."

    # Проверка, что ожидаемая запись найдена
    found = find_entry_in_results(entries, test_case)
    assert found, \
        f"Expected entry not found for: {text[:60]}\n" \
        f"Expected: {test_case.get('expected_wrong') or test_case.get('expected_name') or test_case.get('expected_id')}\n" \
        f"Found entries: {[(e.get('wrong'), e.get('name'), e.get('id')) for e in entries[:5]]}"