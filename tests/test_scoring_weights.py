"""Тесты весов скоринга (итерация 16.1 карты №3)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.scoring_weights import (
    _DEFAULT_WEIGHTS,
    get_scoring_weight,
    load_scoring_weights,
    reset_scoring_weights_cache,
)


@pytest.fixture(autouse=True)
def _reset_weights_cache():
    """Сброс lru_cache до и после каждого теста (кеш модульный, процессный)."""
    reset_scoring_weights_cache()
    yield
    reset_scoring_weights_cache()


@pytest.fixture
def weights_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Путь к временному scoring_weights.json; подменяет _CONFIG_PATH."""
    path = tmp_path / "scoring_weights.json"
    monkeypatch.setattr("src.scoring_weights._CONFIG_PATH", path)
    return path


def test_load_missing_file_returns_defaults(
    weights_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Файл отсутствует — дефолты и warning."""
    caplog.set_level(logging.WARNING, logger="src.scoring_weights")
    result = load_scoring_weights()
    assert result == _DEFAULT_WEIGHTS
    assert all(isinstance(v, int) for v in result.values())
    assert any("not found" in r.message for r in caplog.records)


def test_load_valid_file_returns_values(weights_file: Path) -> None:
    """Валидный JSON — значения из файла, только канонические ключи."""
    weights_file.write_text(
        json.dumps(
            {
                "wrong_exact_match": 111,
                "name_exact_match": 222,
                "partial_text_match": 333,
                "tag_primary": 44,
                "tag_primary_bonus": 5,
                "tag_expanded": 6,
                "extra_key_ignored": 999,
            }
        ),
        encoding="utf-8",
    )
    result = load_scoring_weights()
    assert result == {
        "wrong_exact_match": 111,
        "name_exact_match": 222,
        "partial_text_match": 333,
        "tag_primary": 44,
        "tag_primary_bonus": 5,
        "tag_expanded": 6,
    }
    assert "extra_key_ignored" not in result


def test_load_caches_same_object(weights_file: Path) -> None:
    """Второй load без reset возвращает тот же объект."""
    weights_file.write_text(json.dumps(_DEFAULT_WEIGHTS), encoding="utf-8")
    r1 = load_scoring_weights()
    r2 = load_scoring_weights()
    assert r1 is r2


def test_reset_forces_reload_different_object(weights_file: Path) -> None:
    """После reset — новый объект с теми же значениями."""
    weights_file.write_text(json.dumps(_DEFAULT_WEIGHTS), encoding="utf-8")
    r1 = load_scoring_weights()
    reset_scoring_weights_cache()
    r2 = load_scoring_weights()
    assert r1 is not r2
    assert r1 == r2


def test_invalid_json_returns_defaults(
    weights_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Битый JSON — дефолты и error."""
    weights_file.write_text("not a json", encoding="utf-8")
    caplog.set_level(logging.ERROR, logger="src.scoring_weights")
    result = load_scoring_weights()
    assert result == _DEFAULT_WEIGHTS
    assert any("Failed to load" in r.message for r in caplog.records)


def test_missing_key_returns_defaults(weights_file: Path) -> None:
    """Отсутствует один из обязательных ключей — дефолты."""
    weights_file.write_text(json.dumps({"wrong_exact_match": 1}), encoding="utf-8")
    result = load_scoring_weights()
    assert result == _DEFAULT_WEIGHTS


def test_get_unknown_key_returns_zero_and_warns(
    weights_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Неизвестный ключ в get_scoring_weight — 0 и warning."""
    weights_file.write_text(json.dumps(_DEFAULT_WEIGHTS), encoding="utf-8")
    caplog.set_level(logging.WARNING, logger="src.scoring_weights")
    assert get_scoring_weight("__does_not_exist__") == 0
    assert any("не найден" in r.message for r in caplog.records)
