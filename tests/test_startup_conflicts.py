# tests/test_startup_conflicts.py
"""Тесты для startup-валидации конфликтных правил (Итерация 6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.startup_checks import StartupCheckParams, run_startup_checks


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Создаёт временную директорию с минимальными конфигами."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "domains").mkdir()
    (config_dir / "intents").mkdir()
    (config_dir / "overlays").mkdir()

    core_path = config_dir / "core.json"
    core_path.write_text(json.dumps({"role": "test"}), encoding="utf-8")

    return config_dir


def create_domain(config_dir: Path, name: str, **kwargs) -> None:
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
    data = {
        "name": name,
        "instructions": [],
        **kwargs,
    }
    path = config_dir / "intents" / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def create_overlay(config_dir: Path, name: str, **kwargs) -> None:
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
    create_domain(temp_config_dir, "blog")
    create_domain(temp_config_dir, "marketing")
    create_intent(temp_config_dir, "analytical")
    create_overlay(
        temp_config_dir,
        "landing",
        conflicts_with=["pressrelease"],
        priority=70,
        suppresses=["pressrelease"],
    )
    create_overlay(
        temp_config_dir, "pressrelease", conflicts_with=["landing"], priority=70
    )

    params = StartupCheckParams(
        allowed_domains={"blog", "marketing"},
        allowed_intents={"neutral", "analytical"},
        allowed_overlays={"landing", "pressrelease"},
        config_path=temp_config_dir,
        kb_path=Path("knowledge_base"),
    )
    run_startup_checks(params)


def test_invalid_reference_raises_error(temp_config_dir: Path) -> None:
    create_domain(temp_config_dir, "blog")
    create_overlay(temp_config_dir, "landing", conflicts_with=["nonexistent"])

    params = StartupCheckParams(
        allowed_domains={"blog"},
        allowed_intents={"neutral"},
        allowed_overlays={"landing"},
        config_path=temp_config_dir,
        kb_path=Path("knowledge_base"),
    )
    with pytest.raises(
        ValueError, match="Invalid conflicts_with reference.*nonexistent"
    ):
        run_startup_checks(params)


def test_self_conflict_raises_error(temp_config_dir: Path) -> None:
    create_domain(temp_config_dir, "blog")
    create_overlay(temp_config_dir, "landing", conflicts_with=["landing"])

    params = StartupCheckParams(
        allowed_domains={"blog"},
        allowed_intents={"neutral"},
        allowed_overlays={"landing"},
        config_path=temp_config_dir,
        kb_path=Path("knowledge_base"),
    )
    with pytest.raises(ValueError, match="Self-conflict.*landing"):
        run_startup_checks(params)


def test_suppression_cycle_raises_error(temp_config_dir: Path) -> None:
    create_domain(temp_config_dir, "blog")
    create_overlay(temp_config_dir, "landing", suppresses=["pressrelease"])
    create_overlay(temp_config_dir, "pressrelease", suppresses=["landing"])

    params = StartupCheckParams(
        allowed_domains={"blog"},
        allowed_intents={"neutral"},
        allowed_overlays={"landing", "pressrelease"},
        config_path=temp_config_dir,
        kb_path=Path("knowledge_base"),
    )
    with pytest.raises(ValueError, match="Suppression cycle"):
        run_startup_checks(params)


def test_equal_priority_without_suppress_raises_error(temp_config_dir: Path) -> None:
    create_domain(temp_config_dir, "blog")
    create_overlay(
        temp_config_dir, "landing", conflicts_with=["pressrelease"], priority=70
    )
    create_overlay(
        temp_config_dir, "pressrelease", conflicts_with=["landing"], priority=70
    )

    params = StartupCheckParams(
        allowed_domains={"blog"},
        allowed_intents={"neutral"},
        allowed_overlays={"landing", "pressrelease"},
        config_path=temp_config_dir,
        kb_path=Path("knowledge_base"),
    )
    with pytest.raises(ValueError, match="Equal priority conflict"):
        run_startup_checks(params)


def test_equal_priority_with_suppress_passes(temp_config_dir: Path) -> None:
    create_domain(temp_config_dir, "blog")
    create_overlay(
        temp_config_dir,
        "landing",
        conflicts_with=["pressrelease"],
        priority=70,
        suppresses=["pressrelease"],
    )
    create_overlay(
        temp_config_dir, "pressrelease", conflicts_with=["landing"], priority=70
    )

    params = StartupCheckParams(
        allowed_domains={"blog"},
        allowed_intents={"neutral"},
        allowed_overlays={"landing", "pressrelease"},
        config_path=temp_config_dir,
        kb_path=Path("knowledge_base"),
    )
    run_startup_checks(params)
