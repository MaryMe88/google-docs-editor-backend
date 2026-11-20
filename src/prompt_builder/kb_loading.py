# src/prompt_builder/kb_loading.py
"""
Загрузка базы знаний.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config_types import KnowledgeBase
from src.kb_manifest_loader import (
    ManifestEntry,
    load_manifest,
    select_files_for_request,
)

logger = logging.getLogger(__name__)


def _load_kb_file(
    path: Path,
    expected_key: str | None = None,
    *,
    use_known_keys: bool = True,
) -> list[dict[str, Any]] | dict[str, Any]:
    if not path.exists():
        logger.warning("KB file not found: %s", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load KB file %s: %s", path, e)
        return []
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        logger.warning(
            "KB file %s has unexpected top-level type %s",
            path,
            type(data).__name__,
        )
        return []
    if expected_key and isinstance(data.get(expected_key), list):
        return data[expected_key]
    if not use_known_keys:
        logger.debug("KB file %s treated as dict block (use_known_keys=False)", path)
        return data
    known_list_keys = (
        "items",
        "examples",
        "techniques",
        "frameworks",
        "templates",
        "common_mistakes",
        "issues",
        "rules",
        "entries",
    )
    for key in known_list_keys:
        value = data.get(key)
        if isinstance(value, list):
            return value
    list_valued_keys = [k for k, v in data.items() if isinstance(v, list)]
    if len(list_valued_keys) == 1:
        return data[list_valued_keys[0]]
    logger.debug("KB file %s treated as dict block (no list key found)", path)
    return data


def _build_fallback_manifest_entry(json_path: Path, kb_path: Path) -> ManifestEntry:
    """Создаёт ManifestEntry для fallback-сценария (kb_manifest.json пуст)."""
    rel_path = json_path.relative_to(kb_path)
    stem = json_path.stem
    if stem in ("stop_words", "nkrj_structure_patterns", "domain_glossary"):
        block_type = "dict"
    else:
        block_type = "list"
    return ManifestEntry(
        file=str(rel_path),
        stage="default",
        load_mode="always",
        tags=[],
        intents=[],
        budget_weight="medium",
        status="active",
        priority=99,
        block_name=None,
        block_type=block_type,
    )


def _select_manifest_entries(
    kb_path: Path,
    manifest: list[ManifestEntry],
    active_tags: set[str] | None,
    intent: str | None,
    *,
    load_all: bool,
) -> list[ManifestEntry]:
    """Возвращает список ManifestEntry для загрузки: manifest / rglob-fallback / select."""
    if not load_all:
        return select_files_for_request(manifest, active_tags or set(), intent)

    if manifest:
        return list(manifest)

    fallback: list[ManifestEntry] = []
    for json_path in kb_path.rglob("*.json"):
        fallback.append(_build_fallback_manifest_entry(json_path, kb_path))
    return fallback


def _resolve_block_key(entry: ManifestEntry) -> str:
    """Вычисляет ключ блока: block_name → path-префикс → stem файла."""
    if entry.block_name:
        return entry.block_name
    if "/" in entry.file:
        return entry.file.split("/")[0]
    return Path(entry.file).stem


def _load_entry_records(entry: ManifestEntry, full_path: Path) -> Any:
    """Загружает records одного entry по типу блока (dict vs list)."""
    if getattr(entry, "block_type", "list") == "dict":
        return _load_kb_file(full_path, expected_key=None, use_known_keys=False)
    return _load_kb_file(
        full_path,
        expected_key=entry.block_name or Path(entry.file).stem,
    )


def _merge_entry_into_blocks(
    block_data: dict[str, Any],
    key: str,
    entry: ManifestEntry,
    records: Any,
) -> None:
    """Мержит records в block_data с обработкой type-conflict (dict vs list)."""
    if isinstance(records, dict):
        if key in block_data and isinstance(block_data[key], dict):
            block_data[key].update(records)
        elif key in block_data:
            logger.warning(
                "KB block '%s' type conflict: existing=%s, new=dict — skipping %s",
                key,
                type(block_data[key]).__name__,
                entry.file,
            )
        else:
            block_data[key] = records
        return

    if key not in block_data:
        block_data[key] = []
    if isinstance(block_data[key], list):
        block_data[key].extend(records)
    else:
        logger.warning(
            "KB block '%s' type conflict: existing=%s, new=list — skipping %s",
            key,
            type(block_data[key]).__name__,
            entry.file,
        )


def _ensure_domain_glossary(kb_path: Path, block_data: dict[str, Any]) -> None:
    """Fallback-загрузка domain_glossary.json, если его нет в block_data."""
    if "domain_glossary" in block_data:
        return
    domain_glossary_path = kb_path / "domain_glossary.json"
    if not domain_glossary_path.exists():
        return
    try:
        data = json.loads(domain_glossary_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            block_data["domain_glossary"] = data
    except (json.JSONDecodeError, OSError) as e:
        # Файл есть, но невалидный JSON или недоступен для чтения —
        # это не критично, домен всё равно может быть подключён
        # через манифест.
        logger.error("Failed to load domain_glossary.json: %s", e)


def load_knowledge_base(
    kb_path: Path,
    active_tags: set[str] | None = None,
    intent: str | None = None,
    *,
    load_all: bool = False,
) -> KnowledgeBase:
    """Загружает KB по manifest; load_all=True — все файлы.

    Args:
        kb_path: Корень базы знаний.
        active_tags: Теги для отбора (load_all=False).
        intent: Интент для отбора.
        load_all: Загрузить все блоки, а не только релевантные.

    Returns:
        KnowledgeBase с блоками данных.
    """
    manifest = load_manifest(kb_path / "kb_manifest.json")
    selected = _select_manifest_entries(kb_path, manifest, active_tags, intent, load_all=load_all)

    block_data: dict[str, Any] = {}
    for entry in selected:
        full_path = kb_path / entry.file
        if not full_path.exists():
            logger.warning("KB file not found: %s", full_path)
            continue
        key = _resolve_block_key(entry)
        records = _load_entry_records(entry, full_path)
        if not records:
            continue
        _merge_entry_into_blocks(block_data, key, entry, records)

    _ensure_domain_glossary(kb_path, block_data)

    kb = KnowledgeBase()
    for key, records in block_data.items():
        kb.register(key, records)
    logger.info(
        "Loaded KB with %d blocks from manifest (selected %d files)",
        len(block_data),
        len(selected),
    )
    return kb
