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


def _collect_kb_tags(kb_path: Path) -> Set[str]:
    """Собирает все нормализованные теги из всех записей базы знаний (с учётом вложенности)."""
    kb_tags: Set[str] = set()
    if not kb_path.is_dir():
        logger.warning("Knowledge base directory not found: %s", kb_path)
        return kb_tags

    # Список всех JSON-файлов KB, содержащих записи с полем 'tags'
    kb_files = [
        "grammar_errors.json",
        "stylistic_issues.json",
        "logic_issues.json",
        "composition_errors.json",
        "composition_principles.json",
        "local_cohesion.json",
        "editorial_techniques.json",
        "marketing_templates.json",
        "storytelling_frameworks.json",
        "rhetoric_frameworks.json",
        # stop_words и nkrj не содержат тегов, пропускаем
    ]

    for file_name in kb_files:
        file_path = kb_path / file_name
        if not file_path.is_file():
            logger.warning("KB file not found: %s", file_path)
            continue
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load KB file {file_name}: {e}") from e

        # Структура может быть разной: для grammar_errors — список, для editorial_techniques — список с вложенностью
        items: List[Dict[str, Any]] = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    items.extend(value)

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
    # Извлечь все теги из CANONICAL_TAGS
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
                continue  # ошибки парсинга уже ловятся в check_domain/intent/overlay_files
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

    При несоответствии поднимает RuntimeError с перечнем «висячих» тегов.
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


def run_startup_checks(
    allowed_domains: Set[str],
    allowed_intents: Set[str],
    allowed_overlays: Set[str],
    config_path: Path = Path("config"),
    kb_path: Path = Path("knowledge_base"),
) -> None:
    """
    Выполняет все проверки SYNC-1:
    - домены, интенты, оверлеи → файлы существуют и корректны
    - теги из CANONICAL_TAGS присутствуют в базе знаний
    - wanted_tags из конфигов присутствуют в базе знаний
    """
    logger.info("Running startup checks (SYNC-1)...")
    _check_domain_files(config_path, allowed_domains)
    _check_intent_files(config_path, allowed_intents)
    _check_overlay_files(config_path, allowed_overlays)
    _check_tags_vs_kb(kb_path)                     # существующая проверка
    check_config_tags_vs_kb(config_path, kb_path)  # новая проверка
    logger.info("Startup checks passed successfully.")