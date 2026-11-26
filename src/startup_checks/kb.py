"""KB-проверки: теги, алиасы, соответствие KB.

Итерация 17.2 карты №3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.tag_registry import (
    get_canonical_tag_names,
    normalize_tag,
)

logger = logging.getLogger(__name__)


def _flatten_records(item: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(item, dict):
        nested_keys = [
            k for k in ("examples", "techniques") if k in item and isinstance(item[k], list)
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


_KB_FILES_WITHOUT_TAGS: set[str] = {
    "stop_words.json",
    "nkrj_structure_patterns.json",
}


def _add_normalized_tags(raw_tags: Any, target: set[str]) -> None:
    """Нормализует и добавляет строковые теги в target."""
    if not isinstance(raw_tags, list):
        return
    for tag in raw_tags:
        if isinstance(tag, str):
            norm = normalize_tag(tag)
            if norm:
                target.add(norm)


def _extract_items_and_root_tags(
    data: Any,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Извлекает items для обхода и root-значения tags/inherit_tags.

    Returns:
        (items, root_tags) — items=None означает неподдерживаемый формат.
    """
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict):
        items: list[dict[str, Any]] = []
        root_tags = {k: data.get(k) for k in ("tags", "inherit_tags") if k in data}
        for key, value in data.items():
            if key in ("tags", "inherit_tags"):
                continue
            if isinstance(value, list):
                items.extend(value)
        return items, root_tags
    return None, {}


def _collect_record_tags(items: list[dict[str, Any]], target: set[str]) -> None:
    """Обходит items через _flatten_records и собирает теги из rec["tags"]."""
    for item in items:
        for rec in _flatten_records(item):
            _add_normalized_tags(rec.get("tags"), target)


def _collect_kb_tags(kb_path: Path) -> set[str]:
    """Собирает нормализованные теги из всех KB JSON-файлов (кроме служебных)."""
    kb_tags: set[str] = set()
    if not kb_path.is_dir():
        logger.warning("Knowledge base directory not found: %s", kb_path)
        return kb_tags

    kb_files = sorted(p for p in kb_path.glob("**/*.json") if p.name not in _KB_FILES_WITHOUT_TAGS)
    if not kb_files:
        logger.warning("No KB JSON files found in %s", kb_path)
        return kb_tags

    for file_path in kb_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(f"Failed to load KB file {file_path.name}: {e}") from e

        items, root_tags = _extract_items_and_root_tags(data)
        if items is None:
            logger.warning("Unexpected KB file format in %s, skipping", file_path.name)
            continue
        for raw_tags in root_tags.values():
            _add_normalized_tags(raw_tags, kb_tags)
        _collect_record_tags(items, kb_tags)

    return kb_tags


def _check_tags_vs_kb(kb_path: Path) -> None:
    """Проверяет, что все канонические теги присутствуют в KB (без учёта алиасов)."""
    expected_tags = get_canonical_tag_names()
    expected_tags = {normalize_tag(tag) for tag in expected_tags}

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
    allowed_domains: set[str],
    allowed_intents: set[str],
    allowed_overlays: set[str],
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
