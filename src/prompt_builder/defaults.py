# src/prompt_builder/defaults.py
"""
Дефолтные конфиги и константы, используемые в prompt_builder.
"""

from __future__ import annotations

from src.config_types import DomainConfig, OverlayConfig

# ---------------------------------------------------------------------------
# Константы для валидации kb_limits
# ---------------------------------------------------------------------------
ALLOWED_KB_LIMIT_KEYS: frozenset = frozenset({
    "grammar", "style", "logic", "composition", "cohesion", "local_cohesion",
    "composition_errors", "storytelling", "marketing", "rhetoric", "editorial",
    "glossary", "stop_words", "stop_words_items", "nkrj", "casestudy",
    "grammar_candidates", "style_candidates", "logic_candidates",
    "storytelling_candidates", "marketing_candidates", "rhetoric_candidates",
    "evaluation_techniques",
})

# НОВОЕ: допустимые уровни редактирования (Этап 2)
ALLOWED_EDIT_LEVELS: frozenset = frozenset(
    {"light", "processing", "remake", "adaptive_remake"}
)

# ИЗМЕНЕНИЕ (Итерация 5): разрешаем 0 как допустимое значение для отключения категории
KB_LIMIT_MIN: int = 0
KB_LIMIT_MAX: int = 100


# ---------------------------------------------------------------------------
# Дефолтные конфиги
# ---------------------------------------------------------------------------
_DEFAULT_DOMAIN_CONFIG: DomainConfig = DomainConfig(
    name="general",
    system_rules="",
    tone="neutral",
    allow_storytelling=False,
    allow_marketing=False,
    tasks=(),
    constraints=(),
    ip_ceiling=None,
    kb_limits={},
    priority=100,
    suppresses=(),
    conflicts_with=(),
    incompatible_intents=(),
    incompatible_overlays=(),
)


def _make_default_overlay_config(name: str) -> OverlayConfig:
    return OverlayConfig(
        name=name,
        instructions=(),
        conflicts_with=(),
        priority=70,
        suppresses=(),
    )