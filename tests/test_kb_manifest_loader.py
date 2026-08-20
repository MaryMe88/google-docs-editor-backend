import json
import tempfile
from pathlib import Path

import pytest

from src.kb_manifest_loader import (
    load_manifest,
    select_files_for_request,
    ManifestEntry,
    DEFAULT_MANIFEST_PATH,
)


def test_load_manifest_missing():
    """Если манифест отсутствует, возвращается пустой список."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "missing.json"
        entries = load_manifest(path)
        assert entries == []


def test_load_manifest_valid():
    """Корректный манифест парсится и сортируется по priority."""
    data = {
        "files": [
            {
                "file": "a.json",
                "stage": "stage1",
                "load_mode": "always",
                "tags": ["tag1"],
                "intents": [],
                "budget_weight": "high",
                "status": "active",
                "priority": 2,
            },
            {
                "file": "b.json",
                "stage": "stage1",
                "load_mode": "always",
                "tags": [],
                "intents": [],
                "budget_weight": "medium",
                "status": "active",
                "priority": 1,
            },
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)

    entries = load_manifest(path)
    assert len(entries) == 2
    assert entries[0].file == "b.json"   # priority 1 first
    assert entries[1].file == "a.json"
    Path.unlink(path)


def test_load_manifest_disabled():
    """Записи со статусом disabled пропускаются."""
    data = {
        "files": [
            {
                "file": "disabled.json",
                "stage": "stage1",
                "load_mode": "always",
                "tags": [],
                "intents": [],
                "budget_weight": "low",
                "status": "disabled",
                "priority": 1,
            },
            {
                "file": "active.json",
                "stage": "stage1",
                "load_mode": "always",
                "tags": [],
                "intents": [],
                "budget_weight": "low",
                "status": "active",
                "priority": 2,
            },
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)

    entries = load_manifest(path)
    assert len(entries) == 1
    assert entries[0].file == "active.json"
    Path.unlink(path)


def test_select_files_for_request():
    """Фильтрация по load_mode и пересечению тегов/интентов."""
    manifest = [
        ManifestEntry("a.json", "s1", "always", [], [], "medium", "active", 1),
        ManifestEntry("b.json", "s1", "by_tags", ["tag1"], [], "medium", "active", 2),
        ManifestEntry("c.json", "s1", "by_tags", ["tag2"], [], "medium", "active", 3),
        ManifestEntry("d.json", "s1", "by_intent", [], ["intent1"], "medium", "active", 4),
        ManifestEntry("e.json", "s1", "by_intent", [], ["intent2"], "medium", "active", 5),
    ]

    # active_tags = {"tag1"}, intent = "intent1"
    selected = select_files_for_request(manifest, {"tag1"}, "intent1")
    assert {e.file for e in selected} == {"a.json", "b.json", "d.json"}

    # active_tags = {"tag3"}, intent = None
    selected = select_files_for_request(manifest, {"tag3"}, None)
    assert {e.file for e in selected} == {"a.json"}

    # active_tags = {"tag1", "tag2"}, intent = "intent2"
    selected = select_files_for_request(manifest, {"tag1", "tag2"}, "intent2")
    assert {e.file for e in selected} == {"a.json", "b.json", "c.json", "e.json"}


# ============================================================================
# NEW: Тесты для пилотного реорганизации (case_study)
# ============================================================================

def test_case_study_entry_in_manifest():
    """Проверяет, что манифест содержит запись для genres/business/case_study.json."""
    manifest = load_manifest()
    entry = next((e for e in manifest if e.file == "genres/business/case_study.json"), None)
    assert entry is not None, "Запись для case_study.json не найдена в манифесте"
    assert entry.load_mode == "by_tags"
    assert "casestudy" in entry.tags
    assert entry.status == "active"
    assert entry.block_name == "genre_knowledge" or entry.block_name is not None