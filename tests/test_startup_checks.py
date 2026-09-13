# tests/test_startup_checks.py
"""Тесты для функций проверки при старте (startup_checks.py)."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.startup_checks import (
    StartupCheckParams,
    _check_scoring_weights_file,
    _check_tags_vs_kb,
    run_startup_checks,
)


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

        _check_scoring_weights_file(config_path)


def test_scoring_weights_missing_file_does_not_raise() -> None:
    """Если файл отсутствует, функция не бросает исключение, только предупреждение."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp)
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


def test_prompt_builder_import_and_startup_check() -> None:
    """Проверяет, что PromptBuilder импортируется и startup_check не падает (smoke-тест)."""
    from src.prompt_builder import PromptBuilder

    builder = PromptBuilder()
    builder.startup_check()


def test_check_tags_vs_kb_does_not_warn_for_aliases(tmp_path, caplog) -> None:
    """
    Проверяет, что алиасы (например, story, taiga, antillm) не вызывают предупреждение
    при проверке тегов в KB, даже если они отсутствуют в KB.
    """
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    grammar_file = kb_dir / "grammar_errors.json"
    grammar_file.write_text(
        json.dumps([{"wrong": "test", "correct": "test", "rule": "test", "tags": ["grammar"]}]),
        encoding="utf-8",
    )

    with (
        patch(
            "src.startup_checks.get_canonical_tag_names",
            return_value={"storytelling", "nkrj", "antiai"},
        ),
        caplog.at_level("WARNING"),
    ):
        _check_tags_vs_kb(kb_dir)
        match = re.search(r"missing in KB: \[([^\]]+)\]", caplog.text)
        if match:
            missing_tags_str = match.group(1)
            missing_tags = [tag.strip().strip("'") for tag in missing_tags_str.split(",")]
            assert "story" not in missing_tags
            assert "taiga" not in missing_tags
            assert "antillm" not in missing_tags


def test_check_tags_vs_kb_warns_for_missing_canonical_tag(tmp_path, caplog) -> None:
    """
    Проверяет, что отсутствующий канонический тег в KB вызывает предупреждение.
    """
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    grammar_file = kb_dir / "grammar_errors.json"
    grammar_file.write_text(
        json.dumps([{"wrong": "test", "correct": "test", "rule": "test", "tags": ["grammar"]}]),
        encoding="utf-8",
    )

    with (
        patch("src.startup_checks.get_canonical_tag_names", return_value={"storytelling"}),
        caplog.at_level("WARNING"),
    ):
        _check_tags_vs_kb(kb_dir)
        assert "Tags declared in CANONICAL_TAGS but missing in KB" in caplog.text
        assert "storytelling" in caplog.text


# ---------------------------------------------------------------------------
# ИСПРАВЛЕННЫЙ ТЕСТ: больше не мокаем CANONICAL_TAGS, создаём tag_map.json
# ---------------------------------------------------------------------------
def test_run_startup_checks_does_not_fail_due_to_tag_map() -> None:
    """
    Запуск run_startup_checks не должен падать из-за отсутствия записей в tag_map.json.
    Используем временную папку с минимальными конфигами, чтобы избежать лишних файлов.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "config"
        config_path.mkdir(parents=True)

        # Создаём tag_map.json с пустыми разделами (проверка не упадёт)
        (config_path / "tag_map.json").write_text(
            json.dumps({"domains": {}, "intents": {}, "overlays": {}}), encoding="utf-8"
        )

        domains_dir = config_path / "domains"
        domains_dir.mkdir()
        # Создаём минимальный домен
        (domains_dir / "blog.json").write_text(
            json.dumps(
                {
                    "name": "blog",
                    "system_rules": "",
                    "tone": "neutral",
                    "allow_storytelling": False,
                    "allow_marketing": False,
                }
            ),
            encoding="utf-8",
        )
        # Создаём core.json (может потребоваться для некоторых проверок)
        (config_path / "core.json").write_text(json.dumps({"role": "test"}), encoding="utf-8")
        # Создаём папки intents и overlays (пустые, чтобы не было лишних файлов)
        (config_path / "intents").mkdir()
        (config_path / "overlays").mkdir()

        kb_path = Path(tmp) / "knowledge_base"
        kb_path.mkdir()
        # Создаём минимальный файл KB, чтобы избежать предупреждений
        (kb_path / "grammar_errors.json").write_text(
            json.dumps(
                [
                    {
                        "wrong": "test",
                        "correct": "test",
                        "rule": "test",
                        "tags": ["grammar"],
                    }
                ]
            ),
            encoding="utf-8",
        )

        params = StartupCheckParams(
            allowed_domains={"blog"},
            allowed_intents={"neutral"},
            allowed_overlays=set(),
            config_path=config_path,
            kb_path=kb_path,
        )

        # Запускаем проверки — они не должны упасть (только предупреждения)
        run_startup_checks(params)
        # Если дошли сюда — тест пройден
        assert True
