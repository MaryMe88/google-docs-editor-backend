"""
tests/test_client_sync.py

Проверяет, что все режимы, используемые в Google Apps Script клиенте,
синхронизированы с контрактами бекенда (ALLOWED_DOMAINS, ALLOWED_INTENTS, ALLOWED_OVERLAYS).
"""

from __future__ import annotations

import pytest

from src.shared_contracts import ALLOWED_DOMAINS, ALLOWED_INTENTS, ALLOWED_OVERLAYS


# Это выдержка из MODE_CONFIG клиента (New Script.js)
# Извлечена вручную для проверки.
CLIENT_MODES = [
    # marketing
    {"domain": "marketing", "intent": "analytical", "overlays": ["infostyle"]},
    {"domain": "marketing", "intent": "marketingpush", "overlays": ["infostyle"]},
    {"domain": "marketing", "intent": "analytical", "overlays": ["infostyle"], "output_mode": "text_and_report"},
    # blog
    {"domain": "blog", "intent": "analytical", "overlays": ["infostyle"]},
    {"domain": "blog", "intent": "engagement", "overlays": ["base"]},
    {"domain": "statya", "intent": "analytical", "overlays": ["infostyle", "audiencesegment"]},
    # editing
    {"domain": "basic_edit", "intent": None, "overlays": ["base"]},
    {"domain": "logic_edit", "intent": None, "overlays": ["base"]},
    {"domain": "nora_gal", "intent": None, "overlays": ["base"]},
    {"domain": "nora_gal_soft", "intent": None, "overlays": ["base"]},
    {"domain": "balanced_edit", "intent": None, "overlays": ["base"]},
    {"domain": "deai", "intent": None, "overlays": ["infostyle"], "output_mode": "text_and_report"},
    # cleanup
    {"domain": "readerfirst", "intent": None, "overlays": ["infostyle"]},
    {"domain": "cutnoise", "intent": None, "overlays": ["infostyle"]},
    {"domain": "makeclear", "intent": None, "overlays": ["infostyle"]},
    {"domain": "restructure", "intent": None, "overlays": ["base"]},
    # genre
    {"domain": "genre", "intent": "analytical", "overlays": ["infostyle", "coldemail"]},
    {"domain": "genre", "intent": "analytical", "overlays": ["infostyle", "pressrelease"]},
    {"domain": "genre", "intent": "marketingpush", "overlays": ["infostyle", "landing"]},
    {"domain": "genre", "intent": None, "overlays": ["infostyle", "workdoc"]},
    # creative
    {"domain": "fiction", "intent": "storytelling", "overlays": ["base"]},
    {"domain": "composition", "intent": None, "overlays": ["base"]},
]

# Дополнительно: все оверлеи, которые могут быть использованы (уникальные)
ALL_CLIENT_OVERLAYS = {
    overlay for mode in CLIENT_MODES for overlay in mode["overlays"]
}


def test_all_client_domains_are_allowed() -> None:
    """Все домены из клиентских режимов присутствуют в ALLOWED_DOMAINS."""
    client_domains = {mode["domain"] for mode in CLIENT_MODES}
    missing = client_domains - ALLOWED_DOMAINS
    assert not missing, f"Client domains not in ALLOWED_DOMAINS: {missing}"


def test_all_client_intents_are_allowed() -> None:
    """Все интенты из клиентских режимов (кроме None) присутствуют в ALLOWED_INTENTS."""
    client_intents = {mode["intent"] for mode in CLIENT_MODES if mode["intent"] is not None}
    missing = client_intents - ALLOWED_INTENTS
    assert not missing, f"Client intents not in ALLOWED_INTENTS: {missing}"


def test_all_client_overlays_are_allowed() -> None:
    """Все оверлеи из клиентских режимов присутствуют в ALLOWED_OVERLAYS."""
    missing = ALL_CLIENT_OVERLAYS - ALLOWED_OVERLAYS
    assert not missing, f"Client overlays not in ALLOWED_OVERLAYS: {missing}"