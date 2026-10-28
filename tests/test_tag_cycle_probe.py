"""
tests/test_tag_cycle_probe.py

Характеристические тесты для цикла импорта между:
- src.tag_registry
- src.config_types
- src.config_models.tags

Итерация 6.1 карты №2 — только фиксация текущего поведения.
Runtime не меняется.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from src.config_models.tags import (
    CANONICAL_TAGS,
    KNOWN_TAGS,
    _load_canonical_tags,
    _normalize_tag_local,
    _resolve_normalizers,
    get_primary_tags_for_category,
)
from src.tag_registry import (
    get_canonical_tag_names,
    normalize_tag,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAG_MAP_PATH = PROJECT_ROOT / "config" / "tag_map.json"


# ---------------------------------------------------------------------------
# Группа A: снимки значений (фиксируют "как есть")
# ---------------------------------------------------------------------------


def test_canonical_tags_has_four_categories() -> None:
    """CANONICAL_TAGS содержит 4 категории, включая domains/intents/overlays."""
    assert isinstance(CANONICAL_TAGS, dict)
    assert len(CANONICAL_TAGS) == 4, (
        f"Ожидалось 4 категории, получено {len(CANONICAL_TAGS)}: "
        f"{sorted(CANONICAL_TAGS.keys())}"
    )
    for required in ("domains", "intents", "overlays"):
        assert required in CANONICAL_TAGS, f"отсутствует категория {required}"


def test_canonical_tags_matches_tag_map_file() -> None:
    """CANONICAL_TAGS побайтово совпадает с содержимым config/tag_map.json."""
    assert TAG_MAP_PATH.exists(), f"файл не найден: {TAG_MAP_PATH}"
    with TAG_MAP_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    assert raw == CANONICAL_TAGS


def test_get_canonical_tag_names_returns_41() -> None:
    """get_canonical_tag_names() возвращает 41 тег; basicedit перенумерован в editorial."""
    names = get_canonical_tag_names()
    assert isinstance(names, set)
    assert len(names) == 41, f"Ожидался 41 тег, получено {len(names)}"
    for key_tag in ("editorial", "storytelling"):
        assert key_tag in names, f"отсутствует ключевой тег {key_tag}"
    assert "basicedit" not in names


def test_known_tags_structure() -> None:
    """KNOWN_TAGS — непустое множество строк в нижнем регистре без пробелов."""
    assert isinstance(KNOWN_TAGS, set)
    assert len(KNOWN_TAGS) > 50, f"Ожидалось > 50, получено {len(KNOWN_TAGS)}"
    for tag in KNOWN_TAGS:
        assert isinstance(tag, str)
        assert tag == tag.lower(), f"тег не в нижнем регистре: {tag!r}"
        assert " " not in tag, f"тег содержит пробел: {tag!r}"


def test_normalize_tag_idempotent_for_canonical() -> None:
    """Каждый канонический тег идемпотентен под normalize_tag."""
    for tag in get_canonical_tag_names():
        assert (
            normalize_tag(tag) == tag
        ), f"тег {tag!r} не идемпотентен: normalize_tag → {normalize_tag(tag)!r}"


def test_get_primary_tags_returns_list_for_known_category() -> None:
    """get_primary_tags_for_category возвращает list[str] для известной пары."""
    result = get_primary_tags_for_category("domains", "blog")
    assert isinstance(result, list)
    assert all(isinstance(t, str) for t in result)


# ---------------------------------------------------------------------------
# Группа B: порядки импорта в subprocess
# ---------------------------------------------------------------------------


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Запускает python -c code с PYTHONPATH=PROJECT_ROOT."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=60,
        check=False,
    )


def test_import_tag_registry_standalone() -> None:
    """Изолированный импорт src.tag_registry проходит."""
    result = _run_python("import src.tag_registry")
    assert result.returncode == 0, result.stderr


def test_import_config_types_standalone() -> None:
    """Изолированный импорт src.config_types проходит."""
    result = _run_python("import src.config_types")
    assert result.returncode == 0, result.stderr


def test_import_config_models_tags_standalone() -> None:
    """Изолированный импорт src.config_models.tags проходит."""
    result = _run_python("import src.config_models.tags")
    assert result.returncode == 0, result.stderr


def test_import_forward_order() -> None:
    """Импорт в прямом порядке: tag_registry → config_types → config_models.tags."""
    code = "import src.tag_registry, src.config_types, src.config_models.tags"
    result = _run_python(code)
    assert result.returncode == 0, result.stderr


def test_import_reverse_order() -> None:
    """Импорт в обратном порядке: config_models.tags → config_types → tag_registry."""
    code = "import src.config_models.tags, src.config_types, src.tag_registry"
    result = _run_python(code)
    assert result.returncode == 0, result.stderr


def test_import_then_use_get_canonical_tag_names() -> None:
    """После импорта tag_registry get_canonical_tag_names() возвращает 41."""
    code = (
        "from src.tag_registry import get_canonical_tag_names; "
        "print(len(get_canonical_tag_names()))"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "41", f"stdout={result.stdout!r}"


def test_import_then_use_canonical_tags() -> None:
    """После импорта config_types CANONICAL_TAGS содержит 4 категории."""
    code = "from src.config_types import CANONICAL_TAGS; print(len(CANONICAL_TAGS))"
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "4", f"stdout={result.stdout!r}"


# ---------------------------------------------------------------------------
# Группа C: AST-детекторы top-level обратных импортов
# ---------------------------------------------------------------------------


def _find_module_level_imports_of(path: Path, module_name: str) -> list[tuple[int, str]]:
    """
    Находит импорты `module_name` вне тел функций.

    Возвращает список (lineno, описание). Импорты в теле функций считаются lazy;
    на уровне модуля и в теле класса — top-level (класс исполняется при импорте).
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST, inside_function: bool) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            inside_function = True
        if isinstance(node, ast.ImportFrom):
            if node.module == module_name and not inside_function:
                found.append((node.lineno, f"from {module_name} import ..."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                hit = alias.name == module_name or alias.name.startswith(module_name + ".")
                if hit and not inside_function:
                    found.append((node.lineno, f"import {alias.name}"))
        for child in ast.iter_child_nodes(node):
            visit(child, inside_function)

    visit(tree, False)
    return found


def test_tag_registry_has_no_toplevel_config_types_import() -> None:
    """В tag_registry.py нет top-level импорта src.config_types (только lazy)."""
    path = PROJECT_ROOT / "src" / "tag_registry.py"
    offenders = _find_module_level_imports_of(path, "src.config_types")
    assert not offenders, (
        f"Найден top-level импорт src.config_types в tag_registry.py: {offenders}. "
        "Если это намеренно — удалите тест, потому что цикл разорван."
    )


def test_config_models_tags_has_no_toplevel_tag_registry_import() -> None:
    """В config_models/tags.py нет top-level импорта src.tag_registry (только lazy)."""
    path = PROJECT_ROOT / "src" / "config_models" / "tags.py"
    offenders = _find_module_level_imports_of(path, "src.tag_registry")
    assert not offenders, (
        f"Найден top-level импорт src.tag_registry в config_models/tags.py: {offenders}. "
        "Если это намеренно — удалите тест, потому что цикл разорван."
    )


# ---------------------------------------------------------------------------
# Группа D: fallback-поведение
# ---------------------------------------------------------------------------


def test_load_canonical_tags_empty_on_missing_file(monkeypatch, caplog) -> None:
    """При отсутствии tag_map.json _load_canonical_tags возвращает {} и warning."""
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with caplog.at_level("WARNING"):
        result = _load_canonical_tags()
    assert result == {}
    assert "tag_map.json not found" in caplog.text


def test_load_canonical_tags_empty_on_invalid_json(caplog) -> None:
    """При невалидном JSON _load_canonical_tags возвращает {} и логирует ошибку."""
    with patch.object(Path, "exists", return_value=True):
        with patch(
            "src.config_models.tags.json.load",
            side_effect=json.JSONDecodeError("test", "", 0),
        ):
            with caplog.at_level("ERROR"):
                result = _load_canonical_tags()
    assert result == {}
    assert "Failed to load tag_map.json" in caplog.text


def test_resolve_normalizers_falls_back_when_tag_registry_unavailable(monkeypatch) -> None:
    """При недоступном tag_registry возвращаются локальные фолбэки."""
    monkeypatch.setitem(sys.modules, "src.tag_registry", None)
    normalize_tag_fn, normalize_tags_fn = _resolve_normalizers()
    assert normalize_tag_fn is _normalize_tag_local
    # _normalize_tags_local импортируем для проверки идентичности
    from src.config_models.tags import _normalize_tags_local

    assert normalize_tags_fn is _normalize_tags_local


def test_normalize_tag_local_behavior() -> None:
    """_normalize_tag_local: lowercase, дефисы и пробелы → underscore."""
    assert _normalize_tag_local("Foo-Bar") == "foo_bar"
    assert _normalize_tag_local("FOO BAR") == "foo_bar"
    assert _normalize_tag_local("foo") == "foo"
    assert _normalize_tag_local("UPPER-CASE String") == "upper_case_string"
