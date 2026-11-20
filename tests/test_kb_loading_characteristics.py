"""Характеристические тесты для load_knowledge_base (итерация 18, PR-8).

Фиксируют текущее поведение до рефакторинга:
- fallback rglob при пустом манифесте;
- type-conflict warning (dict vs list);
- domain_glossary fallback вне манифеста;
- missing file warning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from src.prompt_builder import load_knowledge_base


def _write_manifest(kb_dir: Path, entries: list[dict[str, Any]]) -> None:
    (kb_dir / "kb_manifest.json").write_text(json.dumps({"files": entries}), encoding="utf-8")


def _make_entry(
    file: str,
    block_name: str | None = None,
    block_type: str = "list",
) -> dict[str, Any]:
    return {
        "file": file,
        "stage": "default",
        "load_mode": "always",
        "tags": [],
        "intents": [],
        "budget_weight": "medium",
        "status": "active",
        "priority": 99,
        "block_name": block_name,
        "block_type": block_type,
    }


def test_fallback_manifest_when_empty(tmp_path: Path) -> None:
    """Пустой манифест + load_all → rglob; dict-блок по имени stem, остальные как list."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    _write_manifest(kb_dir, [])
    (kb_dir / "stop_words.json").write_text(json.dumps({"w1": 5}), encoding="utf-8")
    (kb_dir / "some_list.json").write_text(
        json.dumps({"some_list": [{"id": "1"}]}), encoding="utf-8"
    )

    kb = load_knowledge_base(kb_dir, load_all=True)

    assert isinstance(kb.get("stop_words"), dict)
    assert isinstance(kb.get("some_list"), list)


def test_type_conflict_dict_then_list_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Первый entry dict, второй list на тот же key → warning, dict побеждает."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    _write_manifest(
        kb_dir,
        [
            _make_entry("a.json", block_name="shared", block_type="dict"),
            _make_entry("b.json", block_name="shared", block_type="list"),
        ],
    )
    (kb_dir / "a.json").write_text(json.dumps({"k": 1}), encoding="utf-8")
    (kb_dir / "b.json").write_text(json.dumps({"entries": [{"id": "1"}]}), encoding="utf-8")

    caplog.set_level(logging.WARNING)
    kb = load_knowledge_base(kb_dir, load_all=True)

    assert kb.get("shared") == {"k": 1}
    assert "type conflict: existing=dict, new=list" in caplog.text


def test_domain_glossary_fallback_when_not_in_manifest(tmp_path: Path) -> None:
    """domain_glossary.json вне манифеста загружается fallback-путём."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    _write_manifest(kb_dir, [_make_entry("grammar.json", block_name="grammar")])
    (kb_dir / "grammar.json").write_text(json.dumps({"entries": [{"id": "1"}]}), encoding="utf-8")
    (kb_dir / "domain_glossary.json").write_text(json.dumps({"term1": "def1"}), encoding="utf-8")

    kb = load_knowledge_base(kb_dir, load_all=True)

    assert kb.get("domain_glossary") == {"term1": "def1"}


def test_missing_file_warns_and_skips(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Entry с несуществующим файлом → warning, не падает, блок отсутствует."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    _write_manifest(kb_dir, [_make_entry("missing.json", block_name="missing")])

    caplog.set_level(logging.WARNING)
    kb = load_knowledge_base(kb_dir, load_all=True)

    assert "KB file not found" in caplog.text
    # KnowledgeBase.get() возвращает [] для отсутствующего ключа (не None).
    assert kb.get("missing") == []
