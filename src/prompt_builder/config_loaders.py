# src/prompt_builder/config_loaders.py
"""
Загрузчики конфигов (JSON-файлы).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.config_types import CoreConfig, DomainConfig, IntentConfig, OverlayConfig
from src.tag_registry import normalize_tag
from .normalization import normalize_string_list
from .defaults import (
    ALLOWED_KB_LIMIT_KEYS,
    ALLOWED_EDIT_LEVELS,
    KB_LIMIT_MIN,
    KB_LIMIT_MAX,
    _DEFAULT_DOMAIN_CONFIG,
    _make_default_overlay_config,
)

logger = logging.getLogger(__name__)


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path, default: Any) -> Any:
    if path.exists():
        return load_json_file(path)
    return default


def load_core_config(base_path: Path = Path("config")) -> CoreConfig:
    data = load_json_file(base_path / "core.json")
    ip_ceiling_raw = data.get("ip_ceiling", {})
    ip_ceiling_value = (
        ip_ceiling_raw.get("value", 2.5)
        if isinstance(ip_ceiling_raw, dict)
        else float(ip_ceiling_raw) if ip_ceiling_raw is not None else 2.5
    )
    return CoreConfig(
        role=data.get("role", "You are a careful Russian editor."),
        priorities=data.get("priorities", "clarity, accuracy, readability"),
        basic_audit_instructions=tuple(data.get("basic_audit_instructions", [])),
        forbidden=tuple(data.get("forbidden", [])),
        ip_ceiling=ip_ceiling_value,
    )


def load_domain_config(domain: str, base_path: Path = Path("config")) -> DomainConfig:
    normalized_domain = domain.strip().lower()
    domain_path = base_path / "domains" / f"{normalized_domain}.json"
    if not domain_path.exists():
        logger.warning(
            "load_domain_config: file not found for domain=%r at %s — "
            "using default domain config.",
            normalized_domain, domain_path,
        )
        return _DEFAULT_DOMAIN_CONFIG
    data = load_json_file(domain_path)
    raw_tasks = data.get("tasks", [])
    raw_constraints = data.get("constraints", [])
    raw_ip = data.get("ip_ceiling")
    domain_ip_ceiling: Optional[float] = None
    if isinstance(raw_ip, (int, float)):
        domain_ip_ceiling = float(raw_ip)
    elif isinstance(raw_ip, dict):
        domain_ip_ceiling = float(raw_ip.get("value", 2.5))

    raw_kb_limits = data.get("kb_limits", {})
    kb_limits: Dict[str, int] = {}
    if isinstance(raw_kb_limits, dict):
        domain_name = data.get("name", normalized_domain)
        for k, v in raw_kb_limits.items():
            if not isinstance(k, str):
                logger.warning("kb_limits[%r] в домене '%s': ключ не строка — пропущен", k, domain_name)
                continue
            if k not in ALLOWED_KB_LIMIT_KEYS:
                logger.warning(
                    "kb_limits: неизвестный ключ '%s' в домене '%s' — проигнорирован. "
                    "Допустимые: %s",
                    k, domain_name, ", ".join(sorted(ALLOWED_KB_LIMIT_KEYS))
                )
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                logger.warning("kb_limits['%s'] в домене '%s': значение %r не число — пропущено", k, domain_name, v)
                continue
            value = int(v)
            clamped = max(KB_LIMIT_MIN, min(KB_LIMIT_MAX, value))
            if clamped != value:
                logger.warning("kb_limits['%s']=%d в домене '%s' вне диапазона [%d, %d] — приведено к %d",
                               k, value, domain_name, KB_LIMIT_MIN, KB_LIMIT_MAX, clamped)
            kb_limits[k] = clamped

    priority = data.get("priority", 100)
    if not isinstance(priority, int):
        priority = 100
    suppresses = tuple(normalize_string_list(data.get("suppresses", [])))
    conflicts_with = tuple(normalize_string_list(data.get("conflicts_with", [])))
    incompatible_intents = tuple(normalize_string_list(data.get("incompatible_intents", [])))
    incompatible_overlays = tuple(normalize_string_list(data.get("incompatible_overlays", [])))

    # ---- edit_level (НОВОЕ) ----
    raw_edit_level = data.get("edit_level", "processing")
    if not isinstance(raw_edit_level, str) or raw_edit_level not in ALLOWED_EDIT_LEVELS:
        logger.warning(
            "load_domain_config: недопустимый edit_level=%r в домене '%s' — "
            "используется 'processing'. Допустимые значения: %s",
            raw_edit_level,
            data.get("name", normalized_domain),
            sorted(ALLOWED_EDIT_LEVELS),
        )
        raw_edit_level = "processing"

    return DomainConfig(
        name=data.get("name", normalized_domain),
        system_rules=data.get("system_rules", ""),
        tone=data.get("tone", "neutral"),
        allow_storytelling=data.get("allow_storytelling", False),
        allow_marketing=data.get("allow_marketing", False),
        tasks=tuple(t for t in raw_tasks if isinstance(t, str)),
        constraints=tuple(c for c in raw_constraints if isinstance(c, str)),
        ip_ceiling=domain_ip_ceiling,
        kb_limits=kb_limits,
        priority=priority,
        suppresses=suppresses,
        conflicts_with=conflicts_with,
        incompatible_intents=incompatible_intents,
        incompatible_overlays=incompatible_overlays,
        edit_level=raw_edit_level,
    )


def load_intent_config(intent: Optional[str], base_path: Path = Path("config")) -> Optional[IntentConfig]:
    if intent is None or intent == "neutral":
        return None
    normalized = normalize_tag(intent)
    intent_path = base_path / "intents" / f"{normalized}.json"
    if not intent_path.exists():
        logger.warning("load_intent_config: file not found for intent=%r — skipping.", normalized)
        return None
    data = load_json_file(intent_path)
    priority = data.get("priority", 50)
    if not isinstance(priority, int):
        priority = 50
    suppresses = tuple(normalize_string_list(data.get("suppresses", [])))
    conflicts_with = tuple(normalize_string_list(data.get("conflicts_with", [])))
    return IntentConfig(
        name=data.get("name", normalized),
        instructions=tuple(data.get("instructions", [])),
        priority=priority,
        suppresses=suppresses,
        conflicts_with=conflicts_with,
    )


def load_overlay_config(overlay: str, base_path: Path = Path("config")) -> OverlayConfig:
    overlay_path = base_path / "overlays" / f"{overlay}.json"
    if not overlay_path.exists():
        logger.warning("load_overlay_config: file not found for overlay=%r — using default.", overlay)
        return _make_default_overlay_config(overlay)
    data = load_json_file(overlay_path)
    priority = data.get("priority", 70)
    if not isinstance(priority, int):
        priority = 70
    suppresses = tuple(normalize_string_list(data.get("suppresses", [])))
    conflicts_with = tuple(normalize_string_list(data.get("conflicts_with", [])))
    return OverlayConfig(
        name=data.get("name", overlay),
        instructions=tuple(data.get("instructions", [])),
        conflicts_with=conflicts_with,
        priority=priority,
        suppresses=suppresses,
    )


def load_overlay_configs(overlays: Sequence[str], base_path: Path = Path("config")) -> List[OverlayConfig]:
    return [load_overlay_config(ov, base_path) for ov in overlays]


def load_output_format(mode: str, base_path: Path = Path("config")) -> str:
    data = load_json_file(base_path / "output_format.json")
    mode_instruction = data.get(mode, data.get("text_only", "Верни только отредактированный текст."))
    global_rules = data.get("global_formatting_rules", {})
    known_keys = {"allowed_formatting"}
    unknown_keys = set(global_rules.keys()) - known_keys
    if unknown_keys:
        logger.warning("load_output_format: ключи %s в 'global_formatting_rules' не используются.", unknown_keys)
    if not global_rules:
        return mode_instruction
    global_parts: List[str] = []
    allowed_formatting = global_rules.get("allowed_formatting", "")
    if allowed_formatting:
        global_parts.append(allowed_formatting)
    if not global_parts:
        return mode_instruction
    return "\n".join(global_parts) + "\n\n" + mode_instruction