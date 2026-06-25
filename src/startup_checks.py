# src/startup_checks.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

from src.config_types import CANONICAL_TAGS
from src.tag_registry import normalize_tag

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
        raise RuntimeError(
            f"Tags missing in knowledge base: {', '.join(sorted(missing_tags))}\n"
            "Each of these tags must appear in at least one KB record (field 'tags')."
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
        raise RuntimeError(
            f"Tags declared in configs but missing in KB: {', '.join(sorted(missing))}. "
            "Each wanted_tag must appear in at least one KB record's 'tags' field."
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
# Основная функция запуска проверок
# ============================================================================
def run_startup_checks(
    allowed_domains: Set[str],
    allowed_intents: Set[str],
    allowed_overlays: Set[str],
    config_path: Path = Path("config"),
    kb_path: Path = Path("knowledge_base"),
) -> None:
    """
    Выполняет все проверки при старте сервиса:
    - домены, интенты, оверлеи → файлы существуют и корректны
    - PR-4 (НП-1): имена оверлеев идемпотентны к normalize_tag
    - теги из CANONICAL_TAGS присутствуют в базе знаний
    - wanted_tags из конфигов присутствуют в базе знаний
    - проверка файла весов скоринга
    - проверка покрытия tag_map.json
    """
    logger.info("Running startup checks...")
    _check_domain_files(config_path, allowed_domains)
    _check_tag_map_coverage(config_path, allowed_domains, allowed_intents, allowed_overlays)
    _check_intent_files(config_path, allowed_intents)
    _check_overlay_files(config_path, allowed_overlays)
    _check_overlay_names_idempotent(allowed_overlays)  # PR-4 (НП-1)
    _check_tags_vs_kb(kb_path)
    check_config_tags_vs_kb(config_path, kb_path)
    _check_scoring_weights_file(config_path)
    logger.info("Startup checks passed successfully.")