"""
tests/test_startup_checks.py

Тесты для функций проверки при старте (startup_checks.py).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.startup_checks import _check_scoring_weights_file


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