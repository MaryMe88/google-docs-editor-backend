"""Тесты для startup-валидации конфликтных правил (Итерация 6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.startup_checks import run_startup_checks
from src.shared_contracts import ALLOWED_DOMAINS, ALLOWED_INTENTS, ALLOWED_OVERLAYS


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Создаёт временную директорию с минимальными конфигами."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Создаём поддиректории
    (config_dir / "domains").mkdir()
    (config_dir / "intents").mkdir()
    (config_dir / "overlays").mkdir()

    # Минимальный core.json (необходим для загрузки, но не используется в проверках конфликтов)
    core_path = config_dir / "core.json"
    core_path.write_text(json.dumps({"role": "test"}), encoding="utf-8")

    return config_dir


def create_domain(config_dir: Path, name: str, **kwargs) -> None:
    """Создаёт файл домена с заданными полями."""
    data = {
        "name": name,
        "system_rules": "",
        "tone": "neutral",
        "allow_storytelling": False,
        "allow_marketing": False,
        **kwargs,
    }
    path = config_dir / "domains" / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def create_intent(config_dir: Path, name: str, **kwargs) -> None:
    """Создаёт файл интента."""
    data = {
        "name": name,
        "instructions": [],
        **kwargs,
    }
    path = config_dir / "intents" / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def create_overlay(config_dir: Path, name: str, **kwargs) -> None:
    """Создаёт файл оверлея."""
    data = {
        "name": name,
        "instructions": [],
        "conflicts_with": [],
        "priority": 70,
        "suppresses": [],
        **kwargs,
    }
    path = config_dir / "overlays" / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_valid_conflict_rules_pass(temp_config_dir: Path) -> None:
    """Корректные конфликтные правила не вызывают ошибок."""
    # Создаём домены
    create_domain(temp_config_dir, "blog")
    create_domain(temp_config_dir, "marketing")
    # Создаём интенты
    create_intent(temp_config_dir, "analytical")
    # Создаём оверлеи
    create_overlay(temp_config_dir, "landing", conflicts_with=["pressrelease"], priority=70)
    create_overlay(temp_config_dir, "pressrelease", conflicts_with=["landing"], priority=70)
    # Добавляем явное подавление, чтобы избежать ошибки equal priority
    create_overlay(temp_config_dir, "landing", conflicts_with=["pressrelease"], priority=70, suppresses=["pressrelease"])
    create_overlay(temp_config_dir, "pressrelease", conflicts_with=["landing"], priority=70)

    allowed_domains = {"blog", "marketing"}
    allowed_intents = {"neutral", "analytical"}
    allowed_overlays = {"landing", "pressrelease"}

    # Должно пройти без ошибок
    run_startup_checks(allowed_domains, allowed_intents, allowed_overlays, config_path=temp_config_dir)


def test_invalid_reference_raises_error(temp_config_dir: Path) -> None:
    """Невалидная ссылка в conflicts_with вызывает ошибку."""
    create_domain(temp_config_dir, "blog")
    # Создаём оверлей с ссылкой на несуществующий оверлей
    create_overlay(temp_config_dir, "landing", conflicts_with=["nonexistent"])

    allowed_domains = {"blog"}
    allowed_intents = {"neutral"}
    allowed_overlays = {"landing"}

    with pytest.raises(ValueError, match="Invalid conflicts_with reference.*nonexistent"):
        run_startup_checks(allowed_domains, allowed_intents, allowed_overlays, config_path=temp_config_dir)


def test_self_conflict_raises_error(temp_config_dir: Path) -> None:
    """Self-conflict (overlay конфликтует с собой) вызывает ошибку."""
    create_domain(temp_config_dir, "blog")
    create_overlay(temp_config_dir, "landing", conflicts_with=["landing"])

    allowed_domains = {"blog"}
    allowed_intents = {"neutral"}
    allowed_overlays = {"landing"}

    with pytest.raises(ValueError, match="Self-conflict.*landing"):
        run_startup_checks(allowed_domains, allowed_intents, allowed_overlays, config_path=temp_config_dir)


def test_suppression_cycle_raises_error(temp_config_dir: Path) -> None:
    """Цикл suppression (A подавляет B, B подавляет A) вызывает ошибку."""
    create_domain(temp_config_dir, "blog")
    create_overlay(temp_config_dir, "landing", suppresses=["pressrelease"])
    create_overlay(temp_config_dir, "pressrelease", suppresses=["landing"])

    allowed_domains = {"blog"}
    allowed_intents = {"neutral"}
    allowed_overlays = {"landing", "pressrelease"}

    with pytest.raises(ValueError, match="Suppression cycle"):
        run_startup_checks(allowed_domains, allowed_intents, allowed_overlays, config_path=temp_config_dir)


def test_equal_priority_without_suppress_raises_error(temp_config_dir: Path) -> None:
    """Конфликтующие оверлеи с равными приоритетами без явного подавления вызывают ошибку."""
    create_domain(temp_config_dir, "blog")
    create_overlay(temp_config_dir, "landing", conflicts_with=["pressrelease"], priority=70)
    create_overlay(temp_config_dir, "pressrelease", conflicts_with=["landing"], priority=70)

    allowed_domains = {"blog"}
    allowed_intents = {"neutral"}
    allowed_overlays = {"landing", "pressrelease"}

    with pytest.raises(ValueError, match="Equal priority conflict"):
        run_startup_checks(allowed_domains, allowed_intents, allowed_overlays, config_path=temp_config_dir)


def test_equal_priority_with_suppress_passes(temp_config_dir: Path) -> None:
    """Конфликтующие оверлеи с равными приоритетами, но с явным подавлением, проходят."""
    create_domain(temp_config_dir, "blog")
    create_overlay(temp_config_dir, "landing", conflicts_with=["pressrelease"], priority=70, suppresses=["pressrelease"])
    create_overlay(temp_config_dir, "pressrelease", conflicts_with=["landing"], priority=70)

    allowed_domains = {"blog"}
    allowed_intents = {"neutral"}
    allowed_overlays = {"landing", "pressrelease"}

    # Должно пройти без ошибок
    run_startup_checks(allowed_domains, allowed_intents, allowed_overlays, config_path=temp_config_dir)