"""Startup-проверки: пакет.

Итерация 17.2 карты №3. Разделено на kb.py, features.py, providers.py.
Публичный API сохранён через реэкспорты в этом __init__.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.registry import check_alias_consistency

from .features import (
    _ASSEMBLY_DISABLED_FEATURES,
    _FEATURE_RESOLUTION_SCENARIOS,
    _REQUIRED_RESOLUTION_KEYS,
    _apply_budget_disables,
    _check_assembly_diagnostics_invariants,
    _check_feature_resolution_invariants,
    _check_trace_block_invariants,
    _run_feature_resolution_scenario,
    _validate_activation_reasons,
    _validate_expected_feature_flags,
)
from .kb import (
    _KB_FILES_WITHOUT_TAGS,
    _add_normalized_tags,
    _check_tag_map_coverage,
    _check_tags_vs_kb,
    _collect_kb_tags,
    _collect_record_tags,
    _extract_items_and_root_tags,
    _flatten_records,
)
from .providers import (
    _check_conflict_rules,
    _check_domain_files_strict,
    _check_equal_priority_conflicts,
    _check_intent_files_strict,
    _check_overlay_files_strict,
    _check_overlay_names_idempotent,
    _check_registry_consistency,
    _check_scoring_weights_file,
    _check_self_conflicts,
    _check_suppression_cycles,
    _is_valid_feature,
    _load_all_configs,
    _normalize_ref_name,
    _normalize_reference,
    _validate_config_references,
    _validate_intent_file,
    _validate_prefixed_reference,
    _validate_reference,
    _validate_references_in_configs,
    _validate_unprefixed_reference,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StartupCheckParams:
    """Параметры для запуска проверок при старте."""

    allowed_domains: set[str]
    allowed_intents: set[str]
    allowed_overlays: set[str]
    config_path: Path = Path("config")
    kb_path: Path = Path("knowledge_base")


def run_startup_checks(params: StartupCheckParams) -> None:
    """
    Выполняет все проверки при старте сервиса.

    Жёсткие проверки (сервис не стартует при ошибке):
    - наличие файлов для всех доменов/интентов/оверлеев (кроме neutral)
    - отсутствие лишних файлов
    - идемпотентность имён оверлеев
    - корректность конфликтных правил (ссылки, циклы, equal priority)
    """
    logger.info("Running startup checks (strict mode)...")

    config_path = params.config_path
    kb_path = params.kb_path

    _check_domain_files_strict(config_path, params.allowed_domains)
    _check_intent_files_strict(config_path, params.allowed_intents)
    _check_overlay_files_strict(config_path, params.allowed_overlays)
    _check_overlay_names_idempotent(params.allowed_overlays)

    _check_conflict_rules(config_path)

    _check_tag_map_coverage(
        config_path,
        params.allowed_domains,
        params.allowed_intents,
        params.allowed_overlays,
    )
    _check_tags_vs_kb(kb_path)
    _check_scoring_weights_file(config_path)

    try:
        _check_registry_consistency()
        warnings = check_alias_consistency()
        for w in warnings:
            logger.warning(w)
        _check_feature_resolution_invariants(config_path)
        _check_assembly_diagnostics_invariants(config_path)
    except Exception as e:
        # Осознанно широкий перехват: это диагностические проверки,
        # их падение не должно ронять старт сервиса.
        logger.warning("Explainability/registry invariants check failed: %s", e)

    logger.info("Startup checks passed successfully.")


__all__ = [
    "_ASSEMBLY_DISABLED_FEATURES",
    "_FEATURE_RESOLUTION_SCENARIOS",
    "_KB_FILES_WITHOUT_TAGS",
    "_REQUIRED_RESOLUTION_KEYS",
    "StartupCheckParams",
    "_add_normalized_tags",
    "_apply_budget_disables",
    "_check_assembly_diagnostics_invariants",
    "_check_conflict_rules",
    "_check_domain_files_strict",
    "_check_equal_priority_conflicts",
    "_check_feature_resolution_invariants",
    "_check_intent_files_strict",
    "_check_overlay_files_strict",
    "_check_overlay_names_idempotent",
    "_check_registry_consistency",
    "_check_scoring_weights_file",
    "_check_self_conflicts",
    "_check_suppression_cycles",
    "_check_tag_map_coverage",
    "_check_tags_vs_kb",
    "_check_trace_block_invariants",
    "_collect_kb_tags",
    "_collect_record_tags",
    "_extract_items_and_root_tags",
    "_flatten_records",
    "_is_valid_feature",
    "_load_all_configs",
    "_normalize_ref_name",
    "_normalize_reference",
    "_run_feature_resolution_scenario",
    "_validate_activation_reasons",
    "_validate_config_references",
    "_validate_expected_feature_flags",
    "_validate_intent_file",
    "_validate_prefixed_reference",
    "_validate_reference",
    "_validate_references_in_configs",
    "_validate_unprefixed_reference",
    "run_startup_checks",
]
