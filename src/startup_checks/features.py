"""Feature resolution и assembly invariants.

Итерация 17.2 карты №3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from src.config_types import AssemblyTrace
from src.reason_codes import ACTIVATION_REASONS, ReasonCode

logger = logging.getLogger(__name__)


_REQUIRED_RESOLUTION_KEYS: tuple[str, ...] = (
    "activated_features",
    "activation_reasons",
    "suppression_reasons",
    "recognized_aliases",
    "ignored_unknown_values",
)


_FEATURE_RESOLUTION_SCENARIOS: list[tuple[str, str | None, list[str], dict[str, bool]]] = [
    (
        "marketing",
        "marketingpush",
        [],
        {
            "storytelling_enabled": False,
            "marketing_enabled": True,
            "antiai_enabled": False,
        },
    ),
    (
        "blog",
        "storytelling",
        [],
        {
            "storytelling_enabled": True,
            "marketing_enabled": False,
            "antiai_enabled": False,
        },
    ),
    ("deai", None, [], {"antiai_enabled": True}),
    ("blog", None, [], {}),
]


def _validate_expected_feature_flags(
    result_dict: dict[str, Any],
    domain: str,
    intent: str | None,
    overlays: list[str],
    expected: dict[str, bool],
) -> None:
    """Пишет warning для каждого несовпавшего флага из expected."""
    for flag, expected_value in expected.items():
        actual = result_dict.get(flag)
        if actual != expected_value:
            logger.warning(
                "Feature resolution invariant violation for "
                "scenario %s/%s/%s: %s = %s, expected %s",
                domain,
                intent,
                overlays,
                flag,
                actual,
                expected_value,
            )


def _validate_activation_reasons(
    result_dict: dict[str, Any],
    domain: str,
    intent: str | None,
    overlays: list[str],
) -> None:
    """Проверяет, что у включённых фич есть reasons и все reasons известны."""
    for flag in (
        "storytelling_enabled",
        "marketing_enabled",
        "antiai_enabled",
        "rhetoric_enabled",
        "nkrj_enabled",
        "editorial_enabled",
    ):
        if not result_dict.get(flag):
            continue
        feature_name = flag.replace("_enabled", "")
        reasons = result_dict.get("activation_reasons", {}).get(feature_name, [])
        if not reasons:
            logger.warning(
                "Feature %s enabled but no activation reasons for scenario %s/%s/%s",
                feature_name,
                domain,
                intent,
                overlays,
            )
        for reason in reasons:
            if reason not in ACTIVATION_REASONS:
                logger.warning(
                    "Unknown activation reason '%s' for feature %s in scenario %s/%s/%s",
                    reason,
                    feature_name,
                    domain,
                    intent,
                    overlays,
                )


def _run_feature_resolution_scenario(
    pb: Any,
    resolve_fn: Any,
    domain: str,
    intent: str | None,
    overlays: list[str],
    expected: dict[str, bool],
) -> None:
    """Прогоняет один сценарий resolve_prompt_features и валидирует результат."""
    domain_config = pb.get_domain_config(domain)
    intent_config = pb.get_intent_config(intent)
    overlay_configs = pb.get_overlay_configs(overlays) if overlays else []
    result_dict = resolve_fn(
        domain=domain,
        intent=intent,
        overlays=overlays,
        domain_config=domain_config,
        intent_config=intent_config,
        overlay_configs=overlay_configs,
    )
    missing = [k for k in _REQUIRED_RESOLUTION_KEYS if k not in result_dict]
    if missing:
        raise RuntimeError(f"{domain}/{intent}: missing keys {missing}")
    _validate_expected_feature_flags(result_dict, domain, intent, overlays, expected)
    _validate_activation_reasons(result_dict, domain, intent, overlays)


def _check_feature_resolution_invariants(config_path: Path) -> None:
    try:
        from src.prompt_builder import PromptBuilder, resolve_prompt_features
    except ImportError as e:
        logger.warning("Cannot import prompt_builder for invariants check: %s", e)
        return

    pb = PromptBuilder(config_path=config_path)
    pb.startup_check()

    for domain, intent, overlays, expected in _FEATURE_RESOLUTION_SCENARIOS:
        try:
            _run_feature_resolution_scenario(
                pb, resolve_prompt_features, domain, intent, overlays, expected
            )
        except Exception as e:
            # Осознанно широкий перехват: это диагностическая проверка,
            # её падение не должно ронять старт сервиса.
            logger.warning(
                "Feature resolution invariant check failed for scenario %s/%s/%s: %s",
                domain,
                intent,
                overlays,
                e,
            )


_ASSEMBLY_DISABLED_FEATURES: tuple[str, ...] = (
    "storytelling",
    "marketing",
    "rhetoric",
    "editorial",
    "nkrj",
)


def _apply_budget_disables(budget: Any, enabled: dict[str, bool]) -> None:
    """Выключает в budget все неактивные фичи из _ASSEMBLY_DISABLED_FEATURES."""
    for feature in _ASSEMBLY_DISABLED_FEATURES:
        if not enabled[feature]:
            budget.disable(feature)


def _check_trace_block_invariants(trace: Any) -> None:
    """Проверяет инварианты диагностики блоков в trace и пишет warning при нарушении."""
    for diag in trace.blocks:
        if diag.included and not diag.eligible:
            logger.warning(
                "Assembly invariant violation: block '%s' included but "
                "not eligible. Reasons: %s",
                diag.name,
                diag.reason_codes,
            )
        if (
            diag.eligible
            and not diag.included
            and ReasonCode.BLOCK_INELIGIBLE_FEATURE_DISABLED not in diag.reason_codes
        ):
            if not diag.reason_codes:
                logger.warning(
                    "Assembly invariant: block '%s' eligible but not "
                    "included without reason. Reasons: %s",
                    diag.name,
                    diag.reason_codes,
                )


def _check_assembly_diagnostics_invariants(config_path: Path) -> None:
    try:
        from src.config_types import KnowledgeLevel
        from src.prompt_builder import (
            KnowledgeBlockRequest,
            KnowledgeBudgetManager,
            PromptBuilder,
            resolve_prompt_features,
        )
        from src.prompt_builder.kb_rendering import _collect_retrieval_tags
    except ImportError as e:
        logger.warning("Cannot import prompt_builder for assembly invariants check: %s", e)
        return

    pb = PromptBuilder(config_path=config_path)
    pb.startup_check()

    try:
        text = "Тестовый текст для проверки сборки блоков."
        domain = "blog"
        intent = None
        overlays: list[str] = []
        domain_config = pb.get_domain_config(domain)
        intent_config = pb.get_intent_config(intent)
        overlay_configs = pb.get_overlay_configs(overlays) if overlays else []

        features = resolve_prompt_features(
            domain=domain,
            intent=intent,
            overlays=overlays,
            domain_config=domain_config,
            intent_config=intent_config,
            overlay_configs=overlay_configs,
        )
        enabled = {
            "storytelling": features["storytelling_enabled"],
            "marketing": features["marketing_enabled"],
            "antiai": features["antiai_enabled"],
            "rhetoric": features["rhetoric_enabled"],
            "nkrj": features["nkrj_enabled"],
            "editorial": features["editorial_enabled"],
        }

        tag_sets = _collect_retrieval_tags(domain, intent, features["effective_overlays"])
        effective_limits = pb._merge_domain_limits(domain_config)
        budget = KnowledgeBudgetManager(token_budget=None).allocate(
            limits=effective_limits,
            level=KnowledgeLevel.FULL,
        )
        _apply_budget_disables(budget, enabled)

        req = KnowledgeBlockRequest(
            text=text,
            primary_tags=tag_sets["primary"],
            expanded_tags=tag_sets["expanded"],
            budget=budget,
            domain=domain,
            intent=intent,
            overlays=features["effective_overlays"],
            include_few_shot=True,
            total_few_shot_used=0,
            few_shot_seed=None,
            limits=effective_limits,
            storytelling_enabled=enabled["storytelling"],
            marketing_enabled=enabled["marketing"],
            antiai_enabled=enabled["antiai"],
            rhetoric_enabled=enabled["rhetoric"],
            nkrj_enabled=enabled["nkrj"],
            editorial_enabled=enabled["editorial"],
            return_trace=True,
        )
        block_result = cast(
            tuple[str, dict[str, Any], int, AssemblyTrace],
            pb._build_knowledge_block(req),
        )
        _, _, _, trace = block_result
        _check_trace_block_invariants(trace)

    except Exception as e:
        # Осознанно широкий перехват: диагностическая проверка,
        # не должна ронять старт сервиса.
        logger.warning("Assembly diagnostics invariant check failed: %s", e)
