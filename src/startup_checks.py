# src/startup_checks.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Optional, Tuple

from src.config_types import (
    CANONICAL_TAGS,
    FeatureResolutionResult,
    AssemblyTrace,
    AssemblyBlockDiagnostics,
    DomainConfig,
    IntentConfig,
    OverlayConfig,
)
from src.reason_codes import ReasonCode, ACTIVATION_REASONS, SUPPRESSION_REASONS
from src.tag_registry import normalize_tag
from src.registry import check_alias_consistency, CANONICAL_FEATURE_ALIASES
from src.prompt_builder import (
    load_domain_config,
    load_intent_config,
    load_overlay_config,
)
from src.shared_contracts import ALLOWED_DOMAINS, ALLOWED_INTENTS, ALLOWED_OVERLAYS

logger = logging.getLogger(__name__)


# ============================================================================
# Строгие проверки наличия и синхронности конфигов
# ============================================================================

def _check_domain_files_strict(
    config_path: Path,
    allowed_domains: Set[str],
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
            except Exception as e:
                raise RuntimeError(f"Failed to parse domain file {domain}.json: {e}") from e

    if missing:
        raise FileNotFoundError(
            f"Missing domain config files: {', '.join(missing)}"
        )

    existing_files = {p.stem for p in domains_dir.glob("*.json") if p.is_file()}
    extra = existing_files - allowed_domains
    if extra:
        raise RuntimeError(
            f"Extra domain config files not in ALLOWED_DOMAINS: {', '.join(sorted(extra))}"
        )


def _check_intent_files_strict(
    config_path: Path,
    allowed_intents: Set[str],
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
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                instructions = data.get("instructions")
                if not isinstance(instructions, list):
                    raise TypeError(
                        f"Intent {intent}: 'instructions' must be a list, got {type(instructions).__name__}"
                    )
                for idx, item in enumerate(instructions):
                    if not isinstance(item, str):
                        raise TypeError(
                            f"Intent {intent}: instructions[{idx}] must be a string, got {type(item).__name__}"
                        )
            except Exception as e:
                raise RuntimeError(f"Invalid intent file {intent}.json: {e}") from e

    if missing:
        raise FileNotFoundError(
            f"Missing intent config files: {', '.join(missing)}"
        )

    existing_files = {p.stem for p in intents_dir.glob("*.json") if p.is_file()}
    extra = existing_files - intents_with_files
    if extra:
        raise RuntimeError(
            f"Extra intent config files not in ALLOWED_INTENTS: {', '.join(sorted(extra))}"
        )


def _check_overlay_files_strict(
    config_path: Path,
    allowed_overlays: Set[str],
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
            except Exception as e:
                raise RuntimeError(f"Failed to parse overlay file {overlay}.json: {e}") from e

    if missing:
        raise FileNotFoundError(
            f"Missing overlay config files: {', '.join(missing)}"
        )

    existing_files = {p.stem for p in overlays_dir.glob("*.json") if p.is_file()}
    extra = existing_files - allowed_overlays
    if extra:
        raise RuntimeError(
            f"Extra overlay config files not in ALLOWED_OVERLAYS: {', '.join(sorted(extra))}"
        )


def _check_overlay_names_idempotent(allowed_overlays: Set[str]) -> None:
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


# ============================================================================
# Проверка scoring_weights (мягкая, не блокирующая)
# ============================================================================

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
            if not isinstance(data[k], (int, float)):
                logger.warning(
                    "Key '%s' in scoring_weights.json has unexpected type %s; "
                    "expected int or float. Using default weights for this key.",
                    k, type(data[k]).__name__,
                )
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in scoring_weights.json: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to validate scoring_weights.json: {e}") from e


# ============================================================================
# Проверка тегов в KB (мягкая, не блокирующая)
# ============================================================================

def _flatten_records(item: Any) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if isinstance(item, dict):
        nested_keys = [
            k for k in ("examples", "techniques")
            if k in item and isinstance(item[k], list)
        ]
        if nested_keys:
            for key in nested_keys:
                for sub in item[key]:
                    records.extend(_flatten_records(sub))
        else:
            records.append(item)
    elif isinstance(item, list):
        for sub in item:
            records.extend(_flatten_records(sub))
    return records


_KB_FILES_WITHOUT_TAGS: Set[str] = {
    "stop_words.json",
    "nkrj_structure_patterns.json",
}


def _collect_kb_tags(kb_path: Path) -> Set[str]:
    kb_tags: Set[str] = set()
    if not kb_path.is_dir():
        logger.warning("Knowledge base directory not found: %s", kb_path)
        return kb_tags

    kb_files = sorted(
        p for p in kb_path.glob("**/*.json")
        if p.name not in _KB_FILES_WITHOUT_TAGS
    )

    if not kb_files:
        logger.warning("No KB JSON files found in %s", kb_path)
        return kb_tags

    for file_path in kb_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load KB file {file_path.name}: {e}") from e

        items: List[Dict[str, Any]] = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for root_field in ("tags", "inherit_tags"):
                root_tags = data.get(root_field)
                if isinstance(root_tags, list):
                    for tag in root_tags:
                        if isinstance(tag, str):
                            norm = normalize_tag(tag)
                            if norm:
                                kb_tags.add(norm)

            for key, value in data.items():
                if key in ("tags", "inherit_tags"):
                    continue
                if isinstance(value, list):
                    items.extend(value)
        else:
            logger.warning("Unexpected KB file format in %s, skipping", file_path.name)
            continue

        for item in items:
            for rec in _flatten_records(item):
                raw_tags = rec.get("tags")
                if not isinstance(raw_tags, list):
                    continue
                for tag in raw_tags:
                    if isinstance(tag, str):
                        norm = normalize_tag(tag)
                        if norm:
                            kb_tags.add(norm)

    return kb_tags


def _check_tags_vs_kb(kb_path: Path) -> None:
    expected_tags: Set[str] = set()
    for category_data in CANONICAL_TAGS.values():
        for tag_data in category_data.values():
            if isinstance(tag_data, dict):
                for key in ("primary", "expanded"):
                    tags_list = tag_data.get(key, [])
                    if isinstance(tags_list, list):
                        for tag in tags_list:
                            if isinstance(tag, str):
                                expected_tags.add(normalize_tag(tag))
            elif isinstance(tag_data, list):
                for tag in tag_data:
                    if isinstance(tag, str):
                        expected_tags.add(normalize_tag(tag))

    if not expected_tags:
        logger.warning("No expected tags found in CANONICAL_TAGS")
        return

    kb_tags = _collect_kb_tags(kb_path)
    missing_tags = expected_tags - kb_tags
    if missing_tags:
        logger.warning(
            "Tags declared in CANONICAL_TAGS but missing in KB: %s. "
            "KB retrieval for these tags will fall back to NEUTRAL stage.",
            sorted(missing_tags),
        )


def _check_tag_map_coverage(
    config_path: Path,
    allowed_domains: Set[str],
    allowed_intents: Set[str],
    allowed_overlays: Set[str],
) -> None:
    from src.config_types import CANONICAL_TAGS
    if not CANONICAL_TAGS:
        logger.warning("CANONICAL_TAGS is empty — skipping tag map coverage check.")
        return

    for entity, allowed in [
        ("domains", allowed_domains),
        ("intents", allowed_intents - {"neutral"}),
        ("overlays", allowed_overlays),
    ]:
        section = CANONICAL_TAGS.get(entity, {})
        missing = [k for k in allowed if k not in section]
        if missing:
            logger.warning(
                "tag_map.json missing entries for %s: %s. "
                "Tag retrieval will use fallback normalization.",
                entity,
                sorted(missing),
            )


# ============================================================================
# Проверки explainability (не блокирующие)
# ============================================================================

def _check_feature_resolution_invariants(config_path: Path) -> None:
    try:
        from src.prompt_builder import PromptBuilder, resolve_prompt_features
    except ImportError as e:
        logger.warning("Cannot import prompt_builder for invariants check: %s", e)
        return

    pb = PromptBuilder(config_path=config_path)
    pb.startup_check()

    scenarios = [
        ("marketing", "marketingpush", [], {"storytelling_enabled": False, "marketing_enabled": True, "antiai_enabled": False}),
        ("blog", "storytelling", [], {"storytelling_enabled": True, "marketing_enabled": False, "antiai_enabled": False}),
        ("deai", None, [], {"antiai_enabled": True}),
        ("blog", None, [], {}),
    ]

    for domain, intent, overlays, expected in scenarios:
        try:
            domain_config = pb.get_domain_config(domain)
            intent_config = pb.get_intent_config(intent)
            overlay_configs = pb.get_overlay_configs(overlays) if overlays else []
            result_dict = resolve_prompt_features(
                domain=domain,
                intent=intent,
                overlays=overlays,
                domain_config=domain_config,
                intent_config=intent_config,
                overlay_configs=overlay_configs,
            )
            assert "activated_features" in result_dict, f"Missing activated_features for {domain}/{intent}"
            assert "activation_reasons" in result_dict, f"Missing activation_reasons for {domain}/{intent}"
            assert "suppression_reasons" in result_dict, f"Missing suppression_reasons for {domain}/{intent}"
            assert "recognized_aliases" in result_dict, f"Missing recognized_aliases for {domain}/{intent}"
            assert "ignored_unknown_values" in result_dict, f"Missing ignored_unknown_values for {domain}/{intent}"

            for flag, expected_value in expected.items():
                actual = result_dict.get(flag)
                if actual != expected_value:
                    logger.warning(
                        "Feature resolution invariant violation for scenario %s/%s/%s: "
                        "%s = %s, expected %s",
                        domain, intent, overlays, flag, actual, expected_value
                    )

            for flag in ["storytelling_enabled", "marketing_enabled", "antiai_enabled",
                         "rhetoric_enabled", "nkrj_enabled", "editorial_enabled"]:
                if result_dict.get(flag):
                    feature_name = flag.replace("_enabled", "")
                    reasons = result_dict.get("activation_reasons", {}).get(feature_name, [])
                    if not reasons:
                        logger.warning(
                            "Feature %s enabled but no activation reasons for scenario %s/%s/%s",
                            feature_name, domain, intent, overlays
                        )
                    for r in reasons:
                        if r not in ACTIVATION_REASONS:
                            logger.warning(
                                "Unknown activation reason '%s' for feature %s in scenario %s/%s/%s",
                                r, feature_name, domain, intent, overlays
                            )

        except Exception as e:
            logger.warning("Feature resolution invariant check failed for scenario %s/%s/%s: %s",
                           domain, intent, overlays, e)


def _check_assembly_diagnostics_invariants(config_path: Path) -> None:
    try:
        from src.prompt_builder import PromptBuilder, _collect_retrieval_tags, KnowledgeBudgetManager, resolve_prompt_features
        from src.config_types import KnowledgeLevel, LimitsConfig
    except ImportError as e:
        logger.warning("Cannot import prompt_builder for assembly invariants check: %s", e)
        return

    pb = PromptBuilder(config_path=config_path)
    pb.startup_check()

    try:
        text = "Тестовый текст для проверки сборки блоков."
        domain = "blog"
        intent = None
        overlays = []
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
        storytelling_enabled = features["storytelling_enabled"]
        marketing_enabled = features["marketing_enabled"]
        antiai_enabled = features["antiai_enabled"]
        rhetoric_enabled = features["rhetoric_enabled"]
        nkrj_enabled = features["nkrj_enabled"]
        editorial_enabled = features["editorial_enabled"]

        tag_sets = _collect_retrieval_tags(domain, intent, features["effective_overlays"])
        effective_limits = pb._merge_domain_limits(domain_config)
        budget = KnowledgeBudgetManager(token_budget=None).allocate(
            limits=effective_limits,
            level=KnowledgeLevel.FULL,
        )
        if not storytelling_enabled:
            budget.disable("storytelling")
        if not marketing_enabled:
            budget.disable("marketing")
        if not rhetoric_enabled:
            budget.disable("rhetoric")
        if not editorial_enabled:
            budget.disable("editorial")
        if not nkrj_enabled:
            budget.disable("nkrj")

        _, _, _, trace = pb._build_knowledge_block(
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
            storytelling_enabled=storytelling_enabled,
            marketing_enabled=marketing_enabled,
            antiai_enabled=antiai_enabled,
            rhetoric_enabled=rhetoric_enabled,
            nkrj_enabled=nkrj_enabled,
            editorial_enabled=editorial_enabled,
            return_trace=True,
        )

        for diag in trace.blocks:
            if diag.included and not diag.eligible:
                logger.warning(
                    "Assembly invariant violation: block '%s' included but not eligible. "
                    "Reasons: %s",
                    diag.name, diag.reason_codes
                )
            if diag.eligible and not diag.included and ReasonCode.BLOCK_INELIGIBLE_FEATURE_DISABLED not in diag.reason_codes:
                if not diag.reason_codes:
                    logger.warning(
                        "Assembly invariant: block '%s' eligible but not included without reason. "
                        "Reasons: %s",
                        diag.name, diag.reason_codes
                    )

    except Exception as e:
        logger.warning("Assembly diagnostics invariant check failed: %s", e)


def _check_registry_consistency() -> None:
    try:
        from src.registry import get_known_intents, get_known_overlays
        from src.shared_contracts import ALLOWED_INTENTS, ALLOWED_OVERLAYS
        registry_intents = get_known_intents()
        registry_overlays = get_known_overlays()
        if registry_intents != ALLOWED_INTENTS:
            logger.warning(
                "Registry intents (%s) do not match shared_contracts ALLOWED_INTENTS (%s)",
                registry_intents, ALLOWED_INTENTS
            )
        if registry_overlays != ALLOWED_OVERLAYS:
            logger.warning(
                "Registry overlays (%s) do not match shared_contracts ALLOWED_OVERLAYS (%s)",
                registry_overlays, ALLOWED_OVERLAYS
            )
    except ImportError as e:
        logger.warning("Cannot import registry for consistency check: %s", e)


# ============================================================================
# НОВОЕ: Проверка конфликтных правил (Итерация 6)
# ============================================================================

def _normalize_reference(ref: str) -> Tuple[Optional[str], str]:
    """
    Разбирает ссылку вида 'intent:name', 'overlay:name', 'feature:name' или просто 'name'.
    Возвращает (тип, имя) или (None, имя) если тип не указан.
    """
    if ':' in ref:
        parts = ref.split(':', 1)
        if len(parts) == 2:
            return parts[0].strip().lower(), parts[1].strip()
    return None, ref.strip()


def _is_valid_feature(feature_name: str) -> bool:
    """Проверяет, что имя фичи присутствует в CANONICAL_FEATURE_ALIASES."""
    return feature_name in CANONICAL_FEATURE_ALIASES


def _check_conflict_rules(config_path: Path) -> None:
    """
    Проверяет все конфликтные правила:
    - существование ссылок
    - отсутствие self-conflict
    - отсутствие циклов suppression
    - equal priority конфликты должны иметь явного победителя
    """
    logger.info("Checking conflict rules...")

    # Загружаем все конфиги, чтобы получить их правила
    domains: Dict[str, DomainConfig] = {}
    for domain_name in ALLOWED_DOMAINS:
        try:
            domains[domain_name] = load_domain_config(domain_name, config_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load domain config for {domain_name}: {e}") from e

    intents: Dict[str, IntentConfig] = {}
    for intent_name in ALLOWED_INTENTS - {"neutral"}:
        try:
            cfg = load_intent_config(intent_name, config_path)
            if cfg is not None:
                intents[intent_name] = cfg
        except Exception as e:
            raise RuntimeError(f"Failed to load intent config for {intent_name}: {e}") from e

    overlays: Dict[str, OverlayConfig] = {}
    for overlay_name in ALLOWED_OVERLAYS:
        try:
            overlays[overlay_name] = load_overlay_config(overlay_name, config_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load overlay config for {overlay_name}: {e}") from e

    # Проверка ссылок на существование
    all_known_names = set(ALLOWED_INTENTS) | set(ALLOWED_OVERLAYS) | set(CANONICAL_FEATURE_ALIASES.keys())
    # Для обратной совместимости: если ссылка без префикса, пытаемся угадать тип
    def validate_reference(ref: str, source: str, field: str) -> None:
        ref_type, name = _normalize_reference(ref)
        if ref_type == "intent":
            if name not in ALLOWED_INTENTS:
                raise ValueError(f"Invalid {field} reference in {source}: 'intent:{name}' does not exist")
        elif ref_type == "overlay":
            if name not in ALLOWED_OVERLAYS:
                raise ValueError(f"Invalid {field} reference in {source}: 'overlay:{name}' does not exist")
        elif ref_type == "feature":
            if not _is_valid_feature(name):
                raise ValueError(f"Invalid {field} reference in {source}: 'feature:{name}' does not exist")
        else:
            # Без префикса: пробуем определить по наличию
            if name in ALLOWED_OVERLAYS:
                # Это overlay, допустимо, но предупредим
                logger.warning(
                    f"Unprefixed overlay reference '{name}' in {source}.{field} – "
                    "consider using 'overlay:{name}' for clarity."
                )
            elif name in ALLOWED_INTENTS:
                logger.warning(
                    f"Unprefixed intent reference '{name}' in {source}.{field} – "
                    "consider using 'intent:{name}' for clarity."
                )
            elif _is_valid_feature(name):
                logger.warning(
                    f"Unprefixed feature reference '{name}' in {source}.{field} – "
                    "consider using 'feature:{name}' for clarity."
                )
            else:
                raise ValueError(f"Invalid {field} reference in {source}: '{name}' not found in any registry")

    # Проверяем suppresses и conflicts_with всех конфигов
    for domain_name, cfg in domains.items():
        for ref in cfg.suppresses:
            validate_reference(ref, f"domain {domain_name}", "suppresses")
        for ref in cfg.conflicts_with:
            validate_reference(ref, f"domain {domain_name}", "conflicts_with")
        for ref in cfg.incompatible_intents:
            validate_reference(ref, f"domain {domain_name}", "incompatible_intents")
        for ref in cfg.incompatible_overlays:
            validate_reference(ref, f"domain {domain_name}", "incompatible_overlays")

    for intent_name, cfg in intents.items():
        for ref in cfg.suppresses:
            validate_reference(ref, f"intent {intent_name}", "suppresses")
        for ref in cfg.conflicts_with:
            validate_reference(ref, f"intent {intent_name}", "conflicts_with")

    for overlay_name, cfg in overlays.items():
        for ref in cfg.suppresses:
            validate_reference(ref, f"overlay {overlay_name}", "suppresses")
        for ref in cfg.conflicts_with:
            validate_reference(ref, f"overlay {overlay_name}", "conflicts_with")

    # Проверка self-conflict: overlay не может конфликтовать с собой
    for overlay_name, cfg in overlays.items():
        for ref in cfg.conflicts_with:
            _, name = _normalize_reference(ref)
            if name == overlay_name:
                raise ValueError(f"Self-conflict in overlay {overlay_name}: conflicts_with contains itself")

    # Проверка циклов suppression (простая: если A подавляет B и B подавляет A)
    # Строим граф suppression для всех сущностей (домены, интенты, оверлеи)
    # Для простоты ограничимся проверкой циклов длины 2 (взаимное подавление)
    suppression_graph: Dict[str, Set[str]] = {}
    for entity, cfg in {**domains, **intents, **overlays}.items():
        # ключ entity должен быть уникальным, но домены, интенты и оверлеи могут иметь одинаковые имена?
        # В разных каталогах могут быть одинаковые имена, но это маловероятно.
        # Для надёжности используем префикс: domain:{name}, intent:{name}, overlay:{name}
        prefix = "domain" if entity in domains else ("intent" if entity in intents else "overlay")
        key = f"{prefix}:{entity}"
        suppression_graph[key] = set()
        for ref in cfg.suppresses:
            _, name = _normalize_reference(ref)
            # определяем тип подавляемой сущности (пока просто по наличию)
            if name in ALLOWED_OVERLAYS:
                target = f"overlay:{name}"
            elif name in ALLOWED_INTENTS:
                target = f"intent:{name}"
            elif _is_valid_feature(name):
                target = f"feature:{name}"
            else:
                # невалидная ссылка уже поймана выше
                continue
            suppression_graph[key].add(target)

    # Проверяем взаимные циклы (A подавляет B и B подавляет A)
    for a, targets in suppression_graph.items():
        for b in targets:
            if b in suppression_graph and a in suppression_graph[b]:
                raise ValueError(f"Suppression cycle detected: {a} suppresses {b} and {b} suppresses {a}")

    # Проверка equal priority конфликтов: для каждой пары overlays, которые конфликтуют,
    # если их приоритеты равны, должен быть явный победитель (т.е. один из них подавляет другой).
    for overlay_name, cfg in overlays.items():
        for ref in cfg.conflicts_with:
            _, conflict_name = _normalize_reference(ref)
            if conflict_name not in overlays:
                continue  # уже проверено выше
            conflict_cfg = overlays[conflict_name]
            # Проверяем, что конфликт взаимный? (не обязательно, но в наших конфигах он взаимный)
            # Проверяем, есть ли явное подавление
            if cfg.priority == conflict_cfg.priority:
                # Проверяем, что один подавляет другого
                has_suppress = (
                    (conflict_name in cfg.suppresses) or
                    (overlay_name in conflict_cfg.suppresses)
                )
                if not has_suppress:
                    raise ValueError(
                        f"Equal priority conflict between overlays '{overlay_name}' and '{conflict_name}' "
                        f"(both priority {cfg.priority}) without explicit suppress rule. "
                        "Add a suppress rule for one of them."
                    )

    logger.info("Conflict rules validation passed.")


# ============================================================================
# Главная функция запуска проверок (обновлена)
# ============================================================================

def run_startup_checks(
    allowed_domains: Set[str],
    allowed_intents: Set[str],
    allowed_overlays: Set[str],
    config_path: Path = Path("config"),
    kb_path: Path = Path("knowledge_base"),
) -> None:
    """
    Выполняет все проверки при старте сервиса.

    Жёсткие проверки (сервис не стартует при ошибке):
    - наличие файлов для всех доменов/интентов/оверлеев (кроме neutral)
    - отсутствие лишних файлов
    - идемпотентность имён оверлеев
    - корректность конфликтных правил (ссылки, циклы, equal priority)
    """
    logger.info("Running startup checks (strict mode)...")

    # Жёсткие проверки
    _check_domain_files_strict(config_path, allowed_domains)
    _check_intent_files_strict(config_path, allowed_intents)
    _check_overlay_files_strict(config_path, allowed_overlays)
    _check_overlay_names_idempotent(allowed_overlays)

    # НОВОЕ: проверка конфликтных правил (жёсткая)
    _check_conflict_rules(config_path)

    # Мягкие проверки (не блокирующие)
    _check_tag_map_coverage(config_path, allowed_domains, allowed_intents, allowed_overlays)
    _check_tags_vs_kb(kb_path)
    _check_scoring_weights_file(config_path)

    # Проверки согласованности и explainability (не блокирующие)
    try:
        _check_registry_consistency()
        warnings = check_alias_consistency()
        for w in warnings:
            logger.warning(w)
        _check_feature_resolution_invariants(config_path)
        _check_assembly_diagnostics_invariants(config_path)
    except Exception as e:
        logger.warning("Explainability/registry invariants check failed: %s", e)

    logger.info("Startup checks passed successfully.")