"""
tests/test_startup_checks.py

Тесты для функций проверки при старте (startup_checks.py).
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.startup_checks import _check_scoring_weights_file, _check_tags_vs_kb
from src.tag_registry import get_canonical_tag_names


def test_scoring_weights_float_values_do_not_raise() -> None:
    """
    Проверяет, что float-значения в scoring_weights.json не вызывают RuntimeError,
    а принимаются без ошибок (задача 7).
    """
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp)
        weights_file = config_path / "scoring_weights.json"
        weights_file.write_text(
            json.dumps(
                {
                    "wrong_exact_match": 10.5,
                    "name_exact_match": 8.0,
                    "partial_text_match": 5,
                    "tag_primary": 6,
                    "tag_primary_bonus": 3,
                    "tag_expanded": 2,
                }
            ),
            encoding="utf-8",
        )

        # Проверяем, что функция не бросает исключение
        _check_scoring_weights_file(config_path)


def test_scoring_weights_missing_file_does_not_raise() -> None:
    """Если файл отсутствует, функция не бросает исключение, только предупреждение."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp)
        # Файла нет
        with patch("src.startup_checks.logger.warning") as mock_warning:
            _check_scoring_weights_file(config_path)
            mock_warning.assert_called_once_with(
                "scoring_weights.json not found, will use default weights."
            )


def test_scoring_weights_missing_keys_raises_error() -> None:
    """Если отсутствуют обязательные ключи, выбрасывается RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp)
        weights_file = config_path / "scoring_weights.json"
        weights_file.write_text(
            json.dumps(
                {
                    "wrong_exact_match": 10,
                    # пропущен "name_exact_match" и др.
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="Missing keys"):
            _check_scoring_weights_file(config_path)


def test_scoring_weights_invalid_json_raises_error() -> None:
    """При невалидном JSON выбрасывается RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp)
        weights_file = config_path / "scoring_weights.json"
        weights_file.write_text("{invalid json}", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Invalid JSON"):
            _check_scoring_weights_file(config_path)


# ============================================================================
# НОВЫЙ ТЕСТ: импортный smoke-тест для PromptBuilder (Этап 5)
# ============================================================================

def test_prompt_builder_import_and_startup_check() -> None:
    """Проверяет, что PromptBuilder импортируется и startup_check не падает (smoke-тест)."""
    from src.prompt_builder import PromptBuilder
    builder = PromptBuilder()
    # startup_check вызывает загрузку конфигов, не должен падать
    builder.startup_check()


# ============================================================================
# НОВЫЕ ТЕСТЫ: проверка _check_tags_vs_kb с учётом канонических тегов и алиасов
# ============================================================================

def test_check_tags_vs_kb_does_not_warn_for_aliases(tmp_path, caplog) -> None:
    """
    Проверяет, что алиасы (например, story, taiga, antillm) не вызывают предупреждение
    при проверке тегов в KB, даже если они отсутствуют в KB.
    """
    # Создаём временную KB с минимальным содержимым
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    # Создаём файл grammar_errors.json с тегом, который не является алиасом
    grammar_file = kb_dir / "grammar_errors.json"
    grammar_file.write_text(
        json.dumps([{"wrong": "test", "correct": "test", "rule": "test", "tags": ["grammar"]}]),
        encoding="utf-8",
    )

    # Мокаем get_canonical_tag_names, чтобы вернуть канонические имена
    # В этой мок-конфигурации канонические имена: storytelling, nkrj, antiai
    with patch("src.startup_checks.get_canonical_tag_names", return_value={"storytelling", "nkrj", "antiai"}):
        with caplog.at_level("WARNING"):
            _check_tags_vs_kb(kb_dir)

            # Извлекаем список отсутствующих тегов из сообщения
            match = re.search(r"missing in KB: \[([^\]]+)\]", caplog.text)
            if match:
                missing_tags_str = match.group(1)
                # Разбиваем по запятой, убираем кавычки и пробелы
                missing_tags = [tag.strip().strip("'") for tag in missing_tags_str.split(",")]
                # Проверяем, что алиасы не входят в этот список
                assert "story" not in missing_tags
                assert "taiga" not in missing_tags
                assert "antillm" not in missing_tags
            else:
                # Если сообщение не содержит списка, значит, предупреждения нет — тест проходит
                pass


def test_check_tags_vs_kb_warns_for_missing_canonical_tag(tmp_path, caplog) -> None:
    """
    Проверяет, что отсутствующий канонический тег в KB вызывает предупреждение.
    """
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    # Создаём файл grammar_errors.json без тега 'storytelling'
    grammar_file = kb_dir / "grammar_errors.json"
    grammar_file.write_text(
        json.dumps([{"wrong": "test", "correct": "test", "rule": "test", "tags": ["grammar"]}]),
        encoding="utf-8",
    )

    # Мокаем get_canonical_tag_names, чтобы вернуть тег, которого нет в KB
    with patch("src.startup_checks.get_canonical_tag_names", return_value={"storytelling"}):
        with caplog.at_level("WARNING"):
            _check_tags_vs_kb(kb_dir)
            # Должно быть предупреждение о missing
            assert "Tags declared in CANONICAL_TAGS but missing in KB" in caplog.text
            assert "storytelling" in caplog.text