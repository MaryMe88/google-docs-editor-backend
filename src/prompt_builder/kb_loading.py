# src/prompt_builder/kb_loading.py
"""
Загрузка базы знаний.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from src.config_types import KnowledgeBase
from src.kb_manifest_loader import load_manifest, select_files_for_request, ManifestEntry

logger = logging.getLogger(__name__)


def _load_kb_file(
    path: Path,
    expected_key: Optional[str] = None,
    use_known_keys: bool = True,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
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
        logger.warning("KB file %s has unexpected top-level type %s", path, type(data).__name__)
        return []
    if expected_key and isinstance(data.get(expected_key), list):
        return data[expected_key]
    if not use_known_keys:
        logger.debug("KB file %s treated as dict block (use_known_keys=False)", path)
        return data
    known_list_keys = ("items", "examples", "techniques", "frameworks", "templates", "common_mistakes", "issues", "rules", "entries")
    for key in known_list_keys:
        value = data.get(key)
        if isinstance(value, list):
            return value
    list_valued_keys = [k for k, v in data.items() if isinstance(v, list)]
    if len(list_valued_keys) == 1:
        return data[list_valued_keys[0]]
    logger.debug("KB file %s treated as dict block (no list key found)", path)
    return data


def load_knowledge_base(
    kb_path: Path,
    active_tags: Optional[Set[str]] = None,
    intent: Optional[str] = None,
    load_all: bool = False,
) -> KnowledgeBase:
    manifest = load_manifest(kb_path / "kb_manifest.json")
    if load_all:
        if manifest:
            selected = list(manifest)
        else:
            selected = []
            for json_path in kb_path.rglob("*.json"):
                rel_path = json_path.relative_to(kb_path)
                stem = json_path.stem
                block_type = "dict" if stem in ("stop_words", "nkrj_structure_patterns", "domain_glossary") else "list"
                entry = ManifestEntry(
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
                selected.append(entry)
    else:
        selected = select_files_for_request(manifest, active_tags or set(), intent)

    block_data: Dict[str, Any] = {}
    for entry in selected:
        full_path = kb_path / entry.file
        if not full_path.exists():
            logger.warning("KB file not found: %s", full_path)
            continue
        if entry.block_name:
            key = entry.block_name
        elif "/" in entry.file:
            key = entry.file.split("/")[0]
        else:
            key = Path(entry.file).stem
        if getattr(entry, "block_type", "list") == "dict":
            records = _load_kb_file(full_path, expected_key=None, use_known_keys=False)
        else:
            records = _load_kb_file(full_path, expected_key=entry.block_name or Path(entry.file).stem)
        if not records:
            continue
        if isinstance(records, dict):
            if key in block_data and isinstance(block_data[key], dict):
                block_data[key].update(records)
            elif key in block_data:
                logger.warning("KB block '%s' type conflict: existing=%s, new=dict — skipping %s",
                               key, type(block_data[key]).__name__, entry.file)
            else:
                block_data[key] = records
        else:
            if key not in block_data:
                block_data[key] = []
            if isinstance(block_data[key], list):
                block_data[key].extend(records)
            else:
                logger.warning("KB block '%s' type conflict: existing=%s, new=list — skipping %s",
                               key, type(block_data[key]).__name__, entry.file)

    if "domain_glossary" not in block_data:
        domain_glossary_path = kb_path / "domain_glossary.json"
        if domain_glossary_path.exists():
            try:
                data = json.loads(domain_glossary_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    block_data["domain_glossary"] = data
            except Exception as e:
                logger.error("Failed to load domain_glossary.json: %s", e)

    kb = KnowledgeBase()
    for key, records in block_data.items():
        kb.register(key, records)
    logger.info("Loaded KB with %d blocks from manifest (selected %d files)", len(block_data), len(selected))
    return kb