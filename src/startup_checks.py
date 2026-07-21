# src/startup_checks.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Optional

from src.config_types import (
    CANONICAL_TAGS,
    FeatureResolutionResult,
    AssemblyTrace,
    AssemblyBlockDiagnostics,
)
from src.reason_codes import ReasonCode, ACTIVATION_REASONS, SUPPRESSION_REASONS
from src.tag_registry import normalize_tag
from src.registry import check_alias_consistency

logger = logging.getLogger(__name__)


def _check_domain_files(
    config_path: Path,
    allowed_domains: Set[str],
) -> None:
    """Проверяет существование и читаемость файлов доменов."""
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


def _check_domain_files_soft(
    config_path: Path,
    allowed_domains: Set[str],
) -> None:
    """Мягкая версия: предупреждает о недостающих файлах доменов вместо исключения.

    Используется в run_startup_checks, чтобы сервис мог стартовать даже при
    удалении конфига домена — load_domain_config вернёт дефолтный конфиг.
    """
    domains_dir = config_path / "domains"
    if not domains_dir.is_dir():
        logger.warning(
            "_check_domain_files_soft: domains directory not found: %s — "
            "all domain requests will use default config.",
            domains_dir,
        )
        return

    for domain in allowed_domains:
        file_path = domains_dir / f"{domain}.json"
        if not file_path.is_file():
            logger.warning(
                "Domain config file missing: %s — "
                "requests for domain=%r will use default domain config (neutral tone).",
                file_path, domain,
            )
        else:
            try:
                with open(file_path, encoding="utf-8") as f:
                    json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to parse domain file {domain}.json: {e}") from e


def _check_intent_files(
    config_path: Path,
    allowed_intents: Set[str],
) -> None:
    """Проверяет существование, структуру instructions для интентов (кроме neutral)."""
    intents_dir = config_path / "intents"
    if not intents_dir.is_dir():
        raise FileNotFoundError(f"Intents directory not found: {intents_dir}")

    for intent in allowed_intents:
        if intent == "neutral":
            continue
        file_path = intents_dir / f"{intent}.json"
        if not file_path.is_file():
            raise FileNotFoundError(f"Intent config file not found: {file_path}")
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


def _check_intent_files_soft(
    config_path: Path,
    allowed_intents: Set[str],
) -> None:
    """Мягкая версия: предупреждает о недостающих файлах интентов вместо исключения.

    Используется в run_startup_checks. При отсутствии файла load_intent_config
    вернёт None (обрабатывается как neutral).
    """
    intents_dir = config_path / "intents"
    if not intents_dir.is_dir():
        logger.warning(
            "_check_intent_files_soft: intents directory not found: %s — "
            "all intent-specific instructions will be skipped.",
            intents_dir,
        )
        return

    for intent in allowed_intents:
        if intent == "neutral":
            continue
        file_path = intents_dir / f"{intent}.json"
        if not file_path.is_file():
            logger.warning(
                "Intent config file missing: %s — "
                "requests with intent=%r will be treated as neutral.",
                file_path, intent,
            )
        else:
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
                            f"Intent {intent}: instructions[{idx}] must be a string, "
                            f"got {type(item).__name__}"
                        )
            except Exception as e:
                raise RuntimeError(f"Invalid intent file {intent}.json: {e}") from e


def _check_overlay_files(
    config_path: Path,
    allowed_overlays: Set[str],
) -> None:
    """Проверяет существование и читаемость файлов оверлеев."""
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


def _check_overlay_files_soft(
    config_path: Path,
    allowed_overlays: Set[str],
) -> None:
    """Мягкая версия: предупреждает о недостающих файлах оверлеев вместо исключения.

    Используется в run_startup_checks. При отсутствии файла load_overlay_config
    вернёт пустой OverlayConfig (без инструкций).
    """
    overlays_dir = config_path / "overlays"
    if not overlays_dir.is_dir():
        logger.warning(
            "_check_overlay_files_soft: overlays directory not found: %s — "
            "all overlay instructions will be skipped.",
            overlays_dir,
        )
        return

    for overlay in allowed_overlays:
        file_path = overlays_dir / f"{overlay}.json"
        if not file_path.is_file():
            logger.warning(
                "Overlay config file missing: %s — "
                "requests with overlay=%r will use empty overlay (no overlay instructions applied).",
                file_path, overlay,
            )
        else:
            try:
                with open(file_path, encoding="utf-8") as f:
                    json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to parse overlay file {overlay}.json: {e}") from e


def _check_overlay_names_idempotent(allowed_overlays: Set[str]) -> None:
    """PR-4 (НП-1): проверяет, что имена файлов оверлеев идемпотентны к normalize_tag.

    Гарантирует, что нормализация в contracts.py (normalize_tag) и проверка
    допустимости в prompt_builder.py работают с одинаковыми строками —
    без скрытых расхождений при добавлении новых оверлеев.

    Пример нарушения: файл 'Final-Check.json' → normalize_tag даст 'final_check',
    а ALLOWED_OVERLAYS будет содержать 'Final-Check' → несовпадение.
    """
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
# ИСПРАВЛЕННАЯ ФУНКЦИЯ (задача 7)
# ============================================================================
def _check_scoring_weights_file(config_path: Path) -> None:
    """
    Проверяет наличие и корректность файла config/scoring_weights.json.
    Если файл отсутствует — только предупреждение (будут использованы значения по умолчанию).
    Если файл присутствует, но повреждён или имеет неверную структуру — ошибка.
    """
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
        # Дополнительные ключи разрешены, но не используются
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in scoring_weights.json: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to validate scoring_weights.json: {e}") from e


def _flatten_records(item: Any) -> List[Dict[str, Any]]:
    """Рекурсивно извлекает все записи с полем 'tags' из item."""
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


# Файлы KB, которые заведомо не содержат поле 'tags' и не нужны для проверки.
# Добавляй сюда новые файлы только если они точно не имеют тегов.
_KB_FILES_WITHOUT_TAGS: Set[str] = {
    "stop_words.json",
    "nkrj_structure_patterns.json",
}


def _collect_kb_tags(kb_path: Path) -> Set[str]:
    """Собирает все нормализованные теги из всех JSON-файлов базы знаний.

    Автоматически сканирует всю папку knowledge_base/ — добавление нового
    файла с тегами не требует правок этого кода.

    Файлы из _KB_FILES_WITHOUT_TAGS пропускаются как заведомо не содержащие
    поле 'tags'.

    Поддерживает два формата KB-файлов:
    - список записей верхнего уровня: ``[{"tags": [...], ...}, ...]``
    - dict с корневыми тегами и вложенными записями:
      ``{"tags": [...], "frameworks": [...]}``
    """
    kb_tags: Set[str] = set()
    if not kb_path.is_dir():
        logger.warning("Knowledge base directory not found: %s", kb_path)
        return kb_tags

    # ИЗМЕНЕНИЕ: рекурсивный обход всех подпапок (заменено *.json на **/*.json)
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
            # Читаем корневые теги (поле "tags" или "inherit_tags") —
            # они применяются ко всему файлу и добавляются напрямую.
            for root_field in ("tags", "inherit_tags"):
                root_tags = data.get(root_field)
                if isinstance(root_tags, list):
                    for tag in root_tags:
                        if isinstance(tag, str):
                            norm = normalize_tag(tag)
                            if norm:
                                kb_tags.add(norm)

            # Все остальные list-значения — это коллекции записей
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


# ============================================================================
# ИЗМЕНЕНИЕ 1.1
# ============================================================================
def _check_tags_vs_kb(kb_path: Path) -> None:
    """
    Собирает все теги, которые могут быть запрошены (из CANONICAL_TAGS),
    и проверяет, что каждый из них присутствует хотя бы в одной записи KB.
    """
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
            "KB retrieval for these tags will fall back to NEUTRAL stage "
            "(rules will be selected without tag matching).",
            sorted(missing_tags),
        )


def _collect_wanted_tags_from_configs(config_path: Path) -> Set[str]:
    """Собирает все wanted_tags из JSON-файлов доменов, интентов и оверлеев."""
    wanted: Set[str] = set()
    for subdir in ("domains", "intents", "overlays"):
        dir_path = config_path / subdir
        if not dir_path.is_dir():
            continue
        for filepath in dir_path.glob("*.json"):
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            raw_tags = data.get("wanted_tags", [])
            if isinstance(raw_tags, list):
                for tag in raw_tags:
                    if isinstance(tag, str):
                        norm = normalize_tag(tag)
                        if norm:
                            wanted.add(norm)
    return wanted


# ============================================================================
# ИЗМЕНЕНИЕ 1.2
# ============================================================================
def check_config_tags_vs_kb(config_path: Path, kb_path: Path) -> None:
    """Проверяет, что каждый wanted_tag из конфигов присутствует
    хотя бы в одной записи базы знаний.
    """
    wanted = _collect_wanted_tags_from_configs(config_path)
    if not wanted:
        logger.warning("check_config_tags_vs_kb: no wanted_tags found in configs, skipping.")
        return
    kb_tags = _collect_kb_tags(kb_path)
    missing = wanted - kb_tags
    if missing:
        logger.warning(
            "wanted_tags declared in configs but missing in KB: %s. "
            "Retrieval for these tags will use fallback (NEUTRAL stage).",
            sorted(missing),
        )


# ============================================================================
# Новая функция для проверки покрытия tag_map.json (задача 3)
# ============================================================================
def _check_tag_map_coverage(
    config_path: Path,
    allowed_domains: Set[str],
    allowed_intents: Set[str],
    allowed_overlays: Set[str],
) -> None:
    """
    Проверяет, что каждый домен/интент/оверлей из allowed_* имеет запись в tag_map.json.
    Если tag_map.json отсутствует — только предупреждение.
    """
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
# NEW: Проверки explainability (третья итерация) — обновлены для четвёртой итерации
# ============================================================================

def _check_feature_resolution_invariants(config_path: Path) -> None:
    """
    Проверяет инварианты feature resolution на representative сценариях.
    Использует реальный PromptBuilder и resolve_prompt_features.
    Теперь использует кэширующие методы PromptBuilder.
    """
    try:
        from src.prompt_builder import PromptBuilder, resolve_prompt_features
        from src.reason_codes import ACTIVATION_REASONS, SUPPRESSION_REASONS
    except ImportError as e:
        logger.warning("Cannot import prompt_builder for invariants check: %s", e)
        return

    pb = PromptBuilder(config_path=config_path)
    pb.startup_check()

    # Сценарии: (domain, intent, overlays, ожидаемые флаги)
    scenarios = [
        ("marketing", "marketingpush", [], {"storytelling_enabled": False, "marketing_enabled": True, "antiai_enabled": False}),
        ("blog", "storytelling", [], {"storytelling_enabled": True, "marketing_enabled": False, "antiai_enabled": False}),
        ("deai", None, [], {"antiai_enabled": True}),
        ("basic_edit", None, ["editorial"], {"editorial_enabled": True}),
        ("blog", None, [], {}),
    ]

    for domain, intent, overlays, expected in scenarios:
        try:
            # Получаем конфиги через кэширующие методы PromptBuilder
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
            # Проверяем наличие диагностических полей
            assert "activated_features" in result_dict, f"Missing activated_features for {domain}/{intent}"
            assert "activation_reasons" in result_dict, f"Missing activation_reasons for {domain}/{intent}"
            assert "suppression_reasons" in result_dict, f"Missing suppression_reasons for {domain}/{intent}"
            assert "recognized_aliases" in result_dict, f"Missing recognized_aliases for {domain}/{intent}"
            assert "ignored_unknown_values" in result_dict, f"Missing ignored_unknown_values for {domain}/{intent}"

            # Проверяем ожидаемые флаги
            for flag, expected_value in expected.items():
                actual = result_dict.get(flag)
                if actual != expected_value:
                    logger.warning(
                        "Feature resolution invariant violation for scenario %s/%s/%s: "
                        "%s = %s, expected %s",
                        domain, intent, overlays, flag, actual, expected_value
                    )

            # Проверяем, что если флаг True, есть activation reason
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
    """
    Проверяет инварианты сборки блоков: eligible/included согласованы с фичами.
    Использует кэширующие методы PromptBuilder.
    """
    try:
        from src.prompt_builder import PromptBuilder, _collect_retrieval_tags, KnowledgeBudgetManager
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


# ============================================================================
# NEW: Проверка согласованности registry и shared_contracts (четвёртая итерация)
# ============================================================================
def _check_registry_consistency() -> None:
    """
    Проверяет, что registry и shared_contracts согласованы.
    Это часть четвёртой итерации для обеспечения единого источника истины.
    """
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
# Основная функция запуска проверок (обновлена для четвёртой итерации)
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
    - директории доменов/интентов/оверлеев существуют
    - существующие файлы конфигов валидны (корректный JSON, нужные поля)
    - имена оверлеев идемпотентны к normalize_tag (PR-4)

    Мягкие проверки (только WARNING, сервис стартует):
    - отдельные файлы доменов/интентов/оверлеев могут отсутствовать —
      load_*_config вернёт дефолтный конфиг
    - теги из CANONICAL_TAGS / wanted_tags могут не покрываться KB —
      retrieval деградирует до NEUTRAL stage
    - scoring_weights.json может отсутствовать — используются дефолтные веса
    - покрытие tag_map.json может быть неполным

    NEW (четвёртая итерация):
    - проверка согласованности registry и shared_contracts
    - все инвариантные проверки используют кэширующие методы PromptBuilder
    """
    logger.info("Running startup checks (fourth iteration)...")

    # Существующие мягкие проверки
    _check_tag_map_coverage(config_path, allowed_domains, allowed_intents, allowed_overlays)
    _check_domain_files_soft(config_path, allowed_domains)
    _check_intent_files_soft(config_path, allowed_intents)
    _check_overlay_files_soft(config_path, allowed_overlays)
    _check_overlay_names_idempotent(allowed_overlays)  # PR-4 (НП-1)
    _check_tags_vs_kb(kb_path)
    check_config_tags_vs_kb(config_path, kb_path)
    _check_scoring_weights_file(config_path)

    # NEW: проверки registry и explainability (четвёртая итерация)
    try:
        _check_registry_consistency()
        # check_alias_consistency импортирована из registry
        warnings = check_alias_consistency()
        for w in warnings:
            logger.warning(w)
        _check_feature_resolution_invariants(config_path)
        _check_assembly_diagnostics_invariants(config_path)
    except Exception as e:
        # Эти проверки не должны останавливать сервис, только логировать
        logger.warning("Explainability/registry invariants check failed: %s", e)

    logger.info("Startup checks passed successfully.")