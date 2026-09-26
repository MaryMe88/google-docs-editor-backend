"""
config_models.explainability

Explainability-структуры для PromptBuilder: результат разрешения фич,
диагностика одного блока и полный трейс сборки knowledge blocks.

Выделено из src.config_types в итерации 7 дорожной карты.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureResolutionResult:
    """
    Канонический результат разрешения фич с explainability.
    Используется внутри prompt_builder и валидации.
    """

    tags: list[str]
    effective_intent: str | None
    effective_overlays: list[str]
    suppressed_layers: list[str]
    warnings: list[str]

    # Feature flags
    storytelling_enabled: bool
    marketing_enabled: bool
    antiai_enabled: bool
    rhetoric_enabled: bool
    nkrj_enabled: bool
    editorial_enabled: bool

    # Explainability
    activated_features: list[str] = field(default_factory=list)
    suppressed_features: list[str] = field(default_factory=list)
    activation_reasons: dict[str, list[str]] = field(default_factory=dict)
    suppression_reasons: dict[str, list[str]] = field(default_factory=dict)
    recognized_aliases: dict[str, list[str]] = field(default_factory=dict)
    ignored_unknown_values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Преобразует в dict для обратной совместимости (используется в build).
        Сохраняет все ключи, которые ожидает существующий код.
        """
        return {
            "tags": self.tags,
            "effective_intent": self.effective_intent,
            "effective_overlays": self.effective_overlays,
            "suppressed_layers": self.suppressed_layers,
            "warnings": self.warnings,
            "storytelling_enabled": self.storytelling_enabled,
            "marketing_enabled": self.marketing_enabled,
            "antiai_enabled": self.antiai_enabled,
            "rhetoric_enabled": self.rhetoric_enabled,
            "nkrj_enabled": self.nkrj_enabled,
            "editorial_enabled": self.editorial_enabled,
            "activated_features": self.activated_features,
            "suppressed_features": self.suppressed_features,
            "activation_reasons": self.activation_reasons,
            "suppression_reasons": self.suppression_reasons,
            "recognized_aliases": self.recognized_aliases,
            "ignored_unknown_values": self.ignored_unknown_values,
        }


@dataclass
class AssemblyBlockDiagnostics:
    """
    Диагностика для одного блока знаний: решение о включении и результат.
    """

    name: str
    eligible: bool
    included: bool
    reason_codes: list[str]
    empty: bool = False
    char_count: int = 0
    entries_count: int = 0


@dataclass
class AssemblyTrace:
    """
    Полная диагностика сборки knowledge blocks.
    Содержит список диагностик для всех блоков и общую статистику.
    """

    blocks: list[AssemblyBlockDiagnostics] = field(default_factory=list)
    total_chars: int = 0
    total_blocks_eligible: int = 0
    total_blocks_included: int = 0
    total_blocks_empty: int = 0

    def add_block(self, diag: AssemblyBlockDiagnostics) -> None:
        self.blocks.append(diag)
        if diag.included:
            self.total_blocks_included += 1
            self.total_chars += diag.char_count
            if diag.empty:
                self.total_blocks_empty += 1
        if diag.eligible:
            self.total_blocks_eligible += 1
