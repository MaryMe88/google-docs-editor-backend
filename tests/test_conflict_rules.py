# tests/test_conflict_rules.py
"""
Unit-тесты для хелперов _check_conflict_rules.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import patch

from src.config_types import DomainConfig, IntentConfig, OverlayConfig
from src.startup_checks import (
    _normalize_ref_name,
    _normalize_reference,
    _is_valid_feature,
    _check_self_conflicts,
    _check_suppression_cycles,
    _check_equal_priority_conflicts,
    _load_all_configs,
    _validate_references_in_configs,
)
from src.registry import CANONICAL_FEATURE_ALIASES


def test_normalize_ref_name():
    assert _normalize_ref_name("overlay:infostyle") == "infostyle"
    assert _normalize_ref_name("intent:analytical") == "analytical"
    assert _normalize_ref_name("simple") == "simple"


def test_normalize_reference():
    assert _normalize_reference("overlay:infostyle") == ("overlay", "infostyle")
    assert _normalize_reference("intent:analytical") == ("intent", "analytical")
    assert _normalize_reference("feature:storytelling") == ("feature", "storytelling")
    assert _normalize_reference("simple") == (None, "simple")


def test_is_valid_feature():
    assert _is_valid_feature("storytelling") is True
    assert _is_valid_feature("nonexistent") is False
    assert _is_valid_feature(list(CANONICAL_FEATURE_ALIASES.keys())[0]) is True


def test_check_self_conflicts_valid():
    overlays = {
        "landing": OverlayConfig(
            name="landing",
            instructions=[],
            priority=70,
            suppresses=[],
            conflicts_with=["pressrelease"],
        )
    }
    _check_self_conflicts(overlays)


def test_check_self_conflicts_invalid():
    overlays = {
        "landing": OverlayConfig(
            name="landing",
            instructions=[],
            priority=70,
            suppresses=[],
            conflicts_with=["landing"],
        )
    }
    with pytest.raises(ValueError, match="Self-conflict"):
        _check_self_conflicts(overlays)


def test_check_suppression_cycles_no_cycle():
    domains = {}
    intents = {}
    overlays = {
        "landing": OverlayConfig(
            name="landing",
            instructions=[],
            priority=70,
            suppresses=["pressrelease"],
            conflicts_with=[],
        ),
        "pressrelease": OverlayConfig(
            name="pressrelease",
            instructions=[],
            priority=70,
            suppresses=[],
            conflicts_with=[],
        ),
    }
    _check_suppression_cycles(domains, intents, overlays)


def test_check_suppression_cycles_cycle():
    domains = {}
    intents = {}
    overlays = {
        "landing": OverlayConfig(
            name="landing",
            instructions=[],
            priority=70,
            suppresses=["pressrelease"],
            conflicts_with=[],
        ),
        "pressrelease": OverlayConfig(
            name="pressrelease",
            instructions=[],
            priority=70,
            suppresses=["landing"],
            conflicts_with=[],
        ),
    }
    with pytest.raises(ValueError, match="Suppression cycle"):
        _check_suppression_cycles(domains, intents, overlays)


def test_check_equal_priority_conflicts_valid():
    overlays = {
        "landing": OverlayConfig(
            name="landing",
            instructions=[],
            priority=70,
            suppresses=["pressrelease"],
            conflicts_with=["pressrelease"],
        ),
        "pressrelease": OverlayConfig(
            name="pressrelease",
            instructions=[],
            priority=70,
            suppresses=[],
            conflicts_with=["landing"],
        ),
    }
    _check_equal_priority_conflicts(overlays)


def test_check_equal_priority_conflicts_invalid():
    overlays = {
        "landing": OverlayConfig(
            name="landing",
            instructions=[],
            priority=70,
            suppresses=[],
            conflicts_with=["pressrelease"],
        ),
        "pressrelease": OverlayConfig(
            name="pressrelease",
            instructions=[],
            priority=70,
            suppresses=[],
            conflicts_with=["landing"],
        ),
    }
    with pytest.raises(ValueError, match="Equal priority conflict"):
        _check_equal_priority_conflicts(overlays)


# ---------------------------------------------------------------------------
# Тесты для _load_all_configs и _validate_references_in_configs
# ---------------------------------------------------------------------------

def test_load_all_configs_success(tmp_path):
    """Проверяет загрузку всех конфигов из временной директории."""
    config_dir = tmp_path / "config"
    domains_dir = config_dir / "domains"
    intents_dir = config_dir / "intents"
    overlays_dir = config_dir / "overlays"
    for d in (domains_dir, intents_dir, overlays_dir):
        d.mkdir(parents=True)

    # Создаём минимальные конфиги
    (domains_dir / "blog.json").write_text(
        json.dumps({"name": "blog", "system_rules": "", "tone": "neutral"}),
        encoding="utf-8"
    )
    (intents_dir / "analytical.json").write_text(
        json.dumps({"name": "analytical", "instructions": []}),
        encoding="utf-8"
    )
    (overlays_dir / "base.json").write_text(
        json.dumps({"name": "base", "instructions": [], "priority": 1, "conflicts_with": [], "suppresses": []}),
        encoding="utf-8"
    )

    # Подменяем ALLOWED_DOMAINS и т.д. на наши временные значения
    with patch("src.startup_checks.ALLOWED_DOMAINS", {"blog"}), \
         patch("src.startup_checks.ALLOWED_INTENTS", {"neutral", "analytical"}), \
         patch("src.startup_checks.ALLOWED_OVERLAYS", {"base"}):
        domains, intents, overlays = _load_all_configs(config_dir)
        assert "blog" in domains
        assert "analytical" in intents
        assert "base" in overlays


def test_validate_references_in_configs_valid(tmp_path):
    """Проверяет валидацию ссылок на существующие сущности."""
    # Создаём домен с корректной ссылкой
    domains = {
        "blog": DomainConfig(
            name="blog",
            system_rules="",
            tone="neutral",
            allow_storytelling=True,
            allow_marketing=False,
            suppresses=["storytelling"],
            conflicts_with=[],
            incompatible_intents=[],
            incompatible_overlays=[],
        )
    }
    intents = {}
    overlays = {}

    # Не должно быть исключения
    _validate_references_in_configs(domains, intents, overlays)

    # Проверяем несуществующую ссылку – создаём новый домен с некорректным suppresses
    domains_bad = {
        "blog": DomainConfig(
            name="blog",
            system_rules="",
            tone="neutral",
            allow_storytelling=True,
            allow_marketing=False,
            suppresses=["nonexistent"],
            conflicts_with=[],
            incompatible_intents=[],
            incompatible_overlays=[],
        )
    }
    with pytest.raises(ValueError, match="Invalid suppresses reference"):
        _validate_references_in_configs(domains_bad, intents, overlays)