from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Set

from src.llm_client import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Путь к папке конфигов (относительно расположения этого файла)
# ---------------------------------------------------------------------------
_CONFIG_BASE = Path(__file__).parent.parent / "config"


def _scan_config_files(subdir: str) -> Set[str]:
    """
    Сканирует папку config/<subdir> и возвращает имена *.json файлов без расширения.
    Если папка не существует, возвращает пустое множество и логирует предупреждение.
    """
    dir_path = _CONFIG_BASE / subdir
    if not dir_path.is_dir():
        logger.warning(f"Config directory not found: {dir_path}")
        return set()
    return {p.stem for p in dir_path.glob("*.json") if p.is_file()}


# ---------------------------------------------------------------------------
# ДОМЕНЫ – автоматически из файлов config/domains/*.json
# ---------------------------------------------------------------------------
ALLOWED_DOMAINS: Final[Set[str]] = _scan_config_files("domains")

# ---------------------------------------------------------------------------
# INTENTS – из файлов config/intents/*.json + служебный "neutral"
# ---------------------------------------------------------------------------
_intents_from_files = _scan_config_files("intents")
# neutral не имеет файла, добавляем вручную
ALLOWED_INTENTS: Final[Set[str]] = _intents_from_files | {"neutral"}

# ---------------------------------------------------------------------------
# OVERLAYS – из файлов config/overlays/*.json
# ---------------------------------------------------------------------------
ALLOWED_OVERLAYS: Final[Set[str]] = _scan_config_files("overlays")

# ---------------------------------------------------------------------------
# Остальные белые списки остаются статическими (не зависят от файлов)
# ---------------------------------------------------------------------------
ALLOWED_OUTPUT_MODES: Final[Set[str]] = {"text_only", "text_and_report"}

# PR-1 (НП-4): ALLOWED_PROVIDERS выводится из LLMProvider — единственный источник правды.
# При добавлении нового провайдера достаточно добавить значение в LLMProvider enum.
ALLOWED_PROVIDERS: Final[Set[str]] = {p.value for p in LLMProvider}

ALLOWED_KIND: Final[Set[str]] = {"b2b", "b2c", "mixed", "custom"}
ALLOWED_EXPERTISE: Final[Set[str]] = {"novice", "pro", "expert"}
ALLOWED_FORMALITY: Final[Set[str]] = {"casual", "neutral", "formal"}
