"""Характеристические тесты для startup_checks strict-проверок (итерация 18, PR-7).

Фиксируют текущее поведение _check_intent_files_strict и _collect_kb_tags
до рефакторинга. Покрывают edge-пути: missing/invalid/extra, неподдерживаемый
формат KB-файла.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.startup_checks import (
    _check_intent_files_strict,
    _collect_kb_tags,
)


def _setup_intents_dir(tmp_path: Path) -> Path:
    """Создаёт config_path/intents/ и возвращает config_path."""
    (tmp_path / "intents").mkdir()
    return tmp_path


# --- _check_intent_files_strict --------------------------------------------


def test_check_intent_files_missing_raises(tmp_path: Path) -> None:
    """Отсутствует файл для allowed intent → FileNotFoundError."""
    config_path = _setup_intents_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing intent config files"):
        _check_intent_files_strict(config_path, {"blog"})


def test_check_intent_files_invalid_json_raises(tmp_path: Path) -> None:
    """Невалидный JSON → RuntimeError с 'Invalid intent file'."""
    config_path = _setup_intents_dir(tmp_path)
    (config_path / "intents" / "blog.json").write_text("not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid intent file blog.json"):
        _check_intent_files_strict(config_path, {"blog"})


def test_check_intent_files_extra_file_raises(tmp_path: Path) -> None:
    """Лишний файл вне allowed_intents → RuntimeError с 'Extra intent config files'."""
    config_path = _setup_intents_dir(tmp_path)
    (config_path / "intents" / "blog.json").write_text(
        json.dumps({"instructions": []}), encoding="utf-8"
    )
    (config_path / "intents" / "extra.json").write_text(
        json.dumps({"instructions": []}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Extra intent config files"):
        _check_intent_files_strict(config_path, {"blog"})


# --- _collect_kb_tags ------------------------------------------------------


def test_collect_kb_tags_unsupported_format_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Файл с корнем-числом → warning, не падает, тег не собирается."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "weird.json").write_text("42", encoding="utf-8")

    caplog.set_level(logging.WARNING)
    result = _collect_kb_tags(kb_dir)

    assert result == set()
    assert "Unexpected KB file format" in caplog.text
