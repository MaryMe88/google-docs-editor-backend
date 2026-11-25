"""Провайдеры конфигов: домены, интенты, оверлеи, конфликты.

Итерация 17.2 карты №3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config_types import DomainConfig, IntentConfig, OverlayConfig
from src.prompt_builder import (
    load_domain_config,
    load_intent_config,
    load_overlay_config,
)
from src.registry import CANONICAL_FEATURE_ALIASES
from src.shared_contracts import (
    ALLOWED_DOMAINS,
    ALLOWED_INTENTS,
    ALLOWED_OVERLAYS,
)
from src.tag_registry import normalize_tag

logger = logging.getLogger(__name__)


def _check_domain_files_strict(
    config_path: Path,
    allowed_domains: set[str],
) -> None:
    """Проверяет, что для каждого домена из ALLOWED_DOMAINS есть файл, и нет лишних файлов."""
    domains_dir = config_path / "domains"
    if not domains_dir.is_dir():
        raise FileNotFoundError(f"Domains directory not found: {domains_dir}")

    missing = []
    for domain in allowed_domains:
        file_path = domains_dir / f"{domain}.json"
        if not file_path.is_file():
            missing.append(domain)
        else:
            try:
                with open(file_path, encoding="utf-8") as f:
                    json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                raise RuntimeError(f"Failed to parse domain file {domain}.json: {e}") from e

    if missing:
        raise FileNotFoundError(f"Missing domain config files: {', '.join(missing)}")

    existing_files = {p.stem for p in domains_dir.glob("*.json") if p.is_file()}
    extra = existing_files - allowed_domains
    if extra:
        raise RuntimeError(
            f"Extra domain config files not in ALLOWED_DOMAINS: " f"{', '.join(sorted(extra))}"
        )


def _validate_intent_file(intent: str, file_path: Path) -> None:
    """Читает intent-файл и валидирует поле instructions (list[str]).

    Raises:
        RuntimeError: JSON невалиден, файл недоступен или instructions некорректны.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        instructions = data.get("instructions")
        if not isinstance(instructions, list):
            raise TypeError(
                f"Intent {intent}: 'instructions' must be a list, "
                f"got {type(instructions).__name__}"
            )
        for idx, item in enumerate(instructions):
            if not isinstance(item, str):
                raise TypeError(
                    f"Intent {intent}: instructions[{idx}] must be a "
                    f"string, got {type(item).__name__}"
                )
    except (json.JSONDecodeError, OSError, TypeError) as e:
        raise RuntimeError(f"Invalid intent file {intent}.json: {e}") from e


def _check_intent_files_strict(
    config_path: Path,
    allowed_intents: set[str],
) -> None:
    """Проверяет файлы интентов, кроме neutral, и отсутствие лишних."""
    intents_dir = config_path / "intents"
    if not intents_dir.is_dir():
        raise FileNotFoundError(f"Intents directory not found: {intents_dir}")

    intents_with_files = allowed_intents - {"neutral"}

    missing = []
    for intent in intents_with_files:
        file_path = intents_dir / f"{intent}.json"
        if not file_path.is_file():
            missing.append(intent)
        else:
            _validate_intent_file(intent, file_path)

    if missing:
        raise FileNotFoundError(f"Missing intent config files: {', '.join(missing)}")

    existing_files = {p.stem for p in intents_dir.glob("*.json") if p.is_file()}
    extra = existing_files - intents_with_files
    if extra:
        raise RuntimeError(
            f"Extra intent config files not in ALLOWED_INTENTS: " f"{', '.join(sorted(extra))}"
        )


def _check_overlay_files_strict(
    config_path: Path,
    allowed_overlays: set[str],
) -> None:
    """Проверяет файлы оверлеев и отсутствие лишних."""
    overlays_dir = config_path / "overlays"
    if not overlays_dir.is_dir():
        raise FileNotFoundError(f"Overlays directory not found: {overlays_dir}")

    missing = []
    for overlay in allowed_overlays:
        file_path = overlays_dir / f"{overlay}.json"
        if not file_path.is_file():
            missing.append(overlay)
        else:
            try:
                with open(file_path, encoding="utf-8") as f:
                    json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                raise RuntimeError(f"Failed to parse overlay file {overlay}.json: {e}") from e

    if missing:
        raise FileNotFoundError(f"Missing overlay config files: {', '.join(missing)}")

    existing_files = {p.stem for p in overlays_dir.glob("*.json") if p.is_file()}
    extra = existing_files - allowed_overlays
    if extra:
        raise RuntimeError(
            f"Extra overlay config files not in ALLOWED_OVERLAYS: " f"{', '.join(sorted(extra))}"
        )


def _check_overlay_names_idempotent(allowed_overlays: set[str]) -> None:
    """PR-4 (НП-1): проверяет, что имена файлов оверлеев идемпотентны к normalize_tag."""
    bad: list[str] = []
    for name in allowed_overlays:
        if normalize_tag(name) != name:
            bad.append(f"'{name}' → normalize_tag → '{normalize_tag(name)}'")
    if bad:
        raise RuntimeError(
            "Overlay filenames are not idempotent to normalize_tag. "
            "Rename the following config/overlays/*.json files so their stem "
            "matches the normalized form:\n" + "\n".join(bad)
        )


def _check_scoring_weights_file(config_path: Path) -> None:
    """Проверяет наличие и корректность файла config/scoring_weights.json."""
    weights_file = config_path / "scoring_weights.json"
    if not weights_file.is_file():
        logger.warning("scoring_weights.json not found, will use default weights.")
        return

    try:
        with open(weights_file, encoding="utf-8") as f:
            data = json.load(f)

        required_keys = {
            "wrong_exact_match",
            "name_exact_match",
            "partial_text_match",
            "tag_primary",
            "tag_primary_bonus",
            "tag_expanded",
        }
        missing = required_keys - data.keys()
        if missing:
            raise RuntimeError(f"Missing keys in scoring_weights.json: {missing}")

        for k in required_keys:
            if not isinstance(data[k], int | float):
                logger.warning(
                    "Key '%s' in scoring_weights.json has unexpected type %s; "
                    "expected int or float. Using default weights for this key.",
                    k,
                    type(data[k]).__name__,
                )
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in scoring_weights.json: {e}") from e
    except (OSError, KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"Failed to validate scoring_weights.json: {e}") from e


def _check_registry_consistency() -> None:
    try:
        from src.registry import get_known_intents, get_known_overlays
        from src.shared_contracts import ALLOWED_INTENTS, ALLOWED_OVERLAYS

        registry_intents = get_known_intents()
        registry_overlays = get_known_overlays()
        if registry_intents != ALLOWED_INTENTS:
            logger.warning(
                "Registry intents (%s) do not match shared_contracts " "ALLOWED_INTENTS (%s)",
                registry_intents,
                ALLOWED_INTENTS,
            )
        if registry_overlays != ALLOWED_OVERLAYS:
            logger.warning(
                "Registry overlays (%s) do not match shared_contracts " "ALLOWED_OVERLAYS (%s)",
                registry_overlays,
                ALLOWED_OVERLAYS,
            )
    except ImportError as e:
        logger.warning("Cannot import registry for consistency check: %s", e)


def _normalize_ref_name(ref: str) -> str:
    """Убирает префиксы overlay: и intent: из ссылки."""
    if ref.startswith("overlay:"):
        return ref[8:]
    if ref.startswith("intent:"):
        return ref[7:]
    return ref


def _normalize_reference(ref: str) -> tuple[str | None, str]:
    """
    Разбирает ссылку вида 'intent:name', 'overlay:name', 'feature:name' или просто 'name'.
    Возвращает (тип, имя) или (None, имя) если тип не указан.
    """
    if ":" in ref:
        parts = ref.split(":", 1)
        if len(parts) == 2:
            return parts[0].strip().lower(), parts[1].strip()
    return None, ref.strip()


def _is_valid_feature(feature_name: str) -> bool:
    """Проверяет, что имя фичи присутствует в CANONICAL_FEATURE_ALIASES."""
    return feature_name in CANONICAL_FEATURE_ALIASES


def _load_all_configs(
    config_path: Path,
) -> tuple[
    dict[str, DomainConfig],
    dict[str, IntentConfig],
    dict[str, OverlayConfig],
]:
    """Загружает все конфиги доменов, интентов и оверлеев."""
    domains: dict[str, DomainConfig] = {}
    for domain_name in ALLOWED_DOMAINS:
        try:
            domains[domain_name] = load_domain_config(domain_name, config_path)
        except (json.JSONDecodeError, OSError, ValueError, TypeError, KeyError) as e:
            raise RuntimeError(f"Failed to load domain config for {domain_name}: {e}") from e

    intents: dict[str, IntentConfig] = {}
    for intent_name in ALLOWED_INTENTS - {"neutral"}:
        try:
            cfg = load_intent_config(intent_name, config_path)
            if cfg is not None:
                intents[intent_name] = cfg
        except (json.JSONDecodeError, OSError, ValueError, TypeError, KeyError) as e:
            raise RuntimeError(f"Failed to load intent config for {intent_name}: {e}") from e

    overlays: dict[str, OverlayConfig] = {}
    for overlay_name in ALLOWED_OVERLAYS:
        try:
            overlays[overlay_name] = load_overlay_config(overlay_name, config_path)
        except (json.JSONDecodeError, OSError, ValueError, TypeError, KeyError) as e:
            raise RuntimeError(f"Failed to load overlay config for {overlay_name}: {e}") from e

    return domains, intents, overlays


def _validate_prefixed_reference(
    ref_type: str,
    name: str,
    source: str,
    field: str,
) -> None:
    """Валидирует prefixed-ссылку (intent:/overlay:/feature:); raise при невалидной."""
    if ref_type == "intent":
        if name not in ALLOWED_INTENTS:
            raise ValueError(
                f"Invalid {field} reference in {source}: 'intent:{name}' does not exist"
            )
        return
    if ref_type == "overlay":
        if name not in ALLOWED_OVERLAYS:
            raise ValueError(
                f"Invalid {field} reference in {source}: 'overlay:{name}' does not exist"
            )
        return
    # ref_type == "feature"
    if not _is_valid_feature(name):
        raise ValueError(f"Invalid {field} reference in {source}: 'feature:{name}' does not exist")


def _validate_unprefixed_reference(name: str, source: str, field: str) -> None:
    """Валидирует unprefixed-ссылку: warning при совпадении с реестром, иначе raise."""
    if name in ALLOWED_OVERLAYS:
        logger.warning(
            f"Unprefixed overlay reference '{name}' in {source}.{field} – "
            f"consider using 'overlay:{name}' for clarity."
        )
    elif name in ALLOWED_INTENTS:
        logger.warning(
            f"Unprefixed intent reference '{name}' in {source}.{field} – "
            f"consider using 'intent:{name}' for clarity."
        )
    elif _is_valid_feature(name):
        logger.warning(
            f"Unprefixed feature reference '{name}' in {source}.{field} – "
            f"consider using 'feature:{name}' for clarity."
        )
    else:
        raise ValueError(
            f"Invalid {field} reference in {source}: '{name}' not found in any registry"
        )


def _validate_reference(ref: str, source: str, field: str) -> None:
    """Валидирует одну ссылку: prefixed или unprefixed."""
    ref_type, name = _normalize_reference(ref)
    if ref_type is None:
        _validate_unprefixed_reference(name, source, field)
    else:
        _validate_prefixed_reference(ref_type, name, source, field)


def _validate_config_references(
    cfg: Any,
    entity_label: str,
    fields: tuple[str, ...],
) -> None:
    """Валидирует все ссылки во всех полях одной конфигурации."""
    for field in fields:
        refs = getattr(cfg, field, ())
        for ref in refs:
            _validate_reference(ref, entity_label, field)


def _validate_references_in_configs(
    domains: dict[str, DomainConfig],
    intents: dict[str, IntentConfig],
    overlays: dict[str, OverlayConfig],
) -> None:
    """Проверяет все ссылки (suppresses, conflicts_with, incompatible_*) на существование."""
    for domain_name, cfg in domains.items():
        _validate_config_references(
            cfg,
            f"domain {domain_name}",
            ("suppresses", "conflicts_with", "incompatible_intents", "incompatible_overlays"),
        )
    for intent_name, intent_cfg in intents.items():
        _validate_config_references(
            intent_cfg,
            f"intent {intent_name}",
            ("suppresses", "conflicts_with"),
        )
    for overlay_name, overlay_cfg in overlays.items():
        _validate_config_references(
            overlay_cfg,
            f"overlay {overlay_name}",
            ("suppresses", "conflicts_with"),
        )


def _check_self_conflicts(overlays: dict[str, OverlayConfig]) -> None:
    """Проверяет, что ни один оверлей не ссылается сам на себя в conflicts_with."""
    for overlay_name, cfg in overlays.items():
        for ref in cfg.conflicts_with:
            _, name = _normalize_reference(ref)
            if name == overlay_name:
                raise ValueError(
                    f"Self-conflict in overlay {overlay_name}: " "conflicts_with contains itself"
                )


def _check_suppression_cycles(
    domains: dict[str, DomainConfig],
    intents: dict[str, IntentConfig],
    overlays: dict[str, OverlayConfig],
) -> None:
    """Строит граф suppression и проверяет на циклы."""
    suppression_graph: dict[str, set[str]] = {}
    combined: dict[str, DomainConfig | IntentConfig | OverlayConfig] = {
        **domains,
        **intents,
        **overlays,
    }
    for entity, cfg in combined.items():
        prefix = "domain" if entity in domains else ("intent" if entity in intents else "overlay")
        key = f"{prefix}:{entity}"
        suppression_graph[key] = set()
        for ref in cfg.suppresses:
            _, name = _normalize_reference(ref)
            if name in ALLOWED_OVERLAYS:
                target = f"overlay:{name}"
            elif name in ALLOWED_INTENTS:
                target = f"intent:{name}"
            elif _is_valid_feature(name):
                target = f"feature:{name}"
            else:
                continue
            suppression_graph[key].add(target)

    for a, targets in suppression_graph.items():
        for b in targets:
            if b in suppression_graph and a in suppression_graph[b]:
                raise ValueError(
                    f"Suppression cycle detected: {a} suppresses {b} " f"and {b} suppresses {a}"
                )


def _check_equal_priority_conflicts(overlays: dict[str, OverlayConfig]) -> None:
    """
    Проверяет, что для каждой пары конфликтующих оверлеев с одинаковым priority
    есть явный suppress в одном из них.
    """
    for overlay_name, cfg in overlays.items():
        for ref in cfg.conflicts_with:
            _, conflict_name = _normalize_reference(ref)
            if conflict_name not in overlays:
                continue
            conflict_cfg = overlays[conflict_name]
            if cfg.priority == conflict_cfg.priority:
                norm_conflict = _normalize_ref_name(conflict_name)
                norm_overlay = _normalize_ref_name(overlay_name)
                has_suppress = any(
                    _normalize_ref_name(s) == norm_conflict for s in cfg.suppresses
                ) or any(_normalize_ref_name(s) == norm_overlay for s in conflict_cfg.suppresses)
                if not has_suppress:
                    raise ValueError(
                        f"Equal priority conflict between overlays "
                        f"'{overlay_name}' and '{conflict_name}' "
                        f"(both priority {cfg.priority}) without explicit "
                        "suppress rule. Add a suppress rule for one of them."
                    )


def _check_conflict_rules(config_path: Path) -> None:
    """
    Проверяет все конфликтные правила:
    - существование ссылок
    - отсутствие self-conflict
    - отсутствие циклов suppression
    - equal priority конфликты должны иметь явного победителя
    """
    logger.info("Checking conflict rules...")

    domains, intents, overlays = _load_all_configs(config_path)
    _validate_references_in_configs(domains, intents, overlays)
    _check_self_conflicts(overlays)
    _check_suppression_cycles(domains, intents, overlays)
    _check_equal_priority_conflicts(overlays)

    logger.info("Conflict rules validation passed.")
