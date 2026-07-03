"""
kb_manifest_loader.py

Читает kb_manifest.json и решает, какие файлы базы знаний загружать
в зависимости от контекста запроса (активные теги и интент).

Манифест становится единственным источником истины о составе KB.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# Путь к манифесту относительно корня проекта (можно переопределить)
DEFAULT_MANIFEST_PATH = Path("knowledge_base/kb_manifest.json")


@dataclass(frozen=True)
class ManifestEntry:
    """Одна запись манифеста, описывающая файл базы знаний."""
    file: str                # путь относительно knowledge_base/
    stage: str               # например "deai_cleanup", "editorial_core"
    load_mode: str           # "always" | "by_tags" | "by_intent" | "never"
    tags: List[str]          # теги для загрузки по совпадению
    intents: List[str]       # интенты для загрузки по совпадению
    budget_weight: str       # "high" | "medium" | "low"
    status: str              # "active" | "disabled"
    priority: int            # порядок загрузки (меньше — раньше)
    block_name: Optional[str] = None  # имя блока для объединения нескольких файлов


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> List[ManifestEntry]:
    """
    Загружает и парсит kb_manifest.json.

    Возвращает список активных записей, отсортированный по priority.
    Если файл не найден или повреждён, возвращает пустой список и логирует ошибку.
    """
    if not path.exists():
        logger.warning("kb_manifest.json not found at %s", path)
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to parse kb_manifest.json: %s", e)
        return []

    entries = []
    for item in data.get("files", []):
        # Пропускаем отключённые записи
        if item.get("status", "active") == "disabled":
            continue

        # Все поля должны присутствовать, но для устойчивости используем .get с дефолтами
        entries.append(ManifestEntry(
            file=item.get("file", ""),
            stage=item.get("stage", ""),
            load_mode=item.get("load_mode", "never"),
            tags=item.get("tags", []),
            intents=item.get("intents", []),
            budget_weight=item.get("budget_weight", "medium"),
            status=item.get("status", "active"),
            priority=item.get("priority", 99),
            block_name=item.get("block_name"),  # новое поле
        ))

    entries.sort(key=lambda e: e.priority)
    logger.info("Loaded %d active entries from kb_manifest.json", len(entries))
    return entries


def select_files_for_request(
    manifest: List[ManifestEntry],
    active_tags: Set[str],
    intent: Optional[str] = None,
) -> List[ManifestEntry]:
    """
    Фильтрует записи манифеста по контексту запроса.

    Правила фильтрации:
      - load_mode == "always" → всегда включается
      - load_mode == "by_tags" → включается, если active_tags пересекается с tags записи
      - load_mode == "by_intent" → включается, если intent совпадает с одним из intents записи
      - load_mode == "never" → не включается (уже отфильтрованы в load_manifest)

    Возвращает список записей, подходящих для загрузки.
    """
    if not manifest:
        return []

    selected = []
    for entry in manifest:
        if entry.load_mode == "always":
            selected.append(entry)
        elif entry.load_mode == "by_tags":
            if active_tags & set(entry.tags):
                selected.append(entry)
        elif entry.load_mode == "by_intent":
            if intent and intent in entry.intents:
                selected.append(entry)
        # load_mode == "never" игнорируем

    logger.debug(
        "Selected %d files for request (intent=%s, tags=%s)",
        len(selected), intent, active_tags
    )
    return selected