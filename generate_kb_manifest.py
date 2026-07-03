#!/usr/bin/env python3
"""
generate_kb_manifest.py — Генерирует kb_manifest.json по реальной структуре
папки knowledge_base.

Использование:
  python generate_kb_manifest.py                     # KB в папке скрипта
  python generate_kb_manifest.py --kb-dir ./knowledge_base
  python generate_kb_manifest.py --dry-run           # показать без записи
  python generate_kb_manifest.py --update            # обновить существующий манифест

Выходной файл: knowledge_base/kb_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ── Константа: имя выходного файла ───────────────────────────────────────────
MANIFEST_FILENAME = "kb_manifest.json"

# ── Таблица правил для каждого файла ─────────────────────────────────────────
# Ключ: имя файла (без пути).
# stage         — смысловой слой в пайплайне
# priority      — 1 (самый ранний) … 9 (самый поздний)
# load_mode     — "always" | "by_tags" | "by_intent"
# budget_weight — "high" | "medium" | "low"
# tags          — теги для retrieval (пустой список = не участвует в tag-поиске)
# intents       — интенты, при которых файл подключается (only for by_intent)
# block_name    — (опционально) имя блока для объединения нескольких файлов

FILE_RULES: dict[str, dict[str, Any]] = {

    # ── deai_cleanup (priority 1) ──────────────────────────────────────────
    "anti_ai_techniques.json": {
        "stage": "deai_cleanup",
        "priority": 1,
        "load_mode": "always",
        "budget_weight": "high",
        "tags": ["anti_ai", "humanize", "deai", "authenticity"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "fighting_officialese.json": {
        "stage": "deai_cleanup",
        "priority": 1,
        "load_mode": "always",
        "budget_weight": "high",
        "tags": ["officialese", "bureaucratic", "deai", "humanize"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "anti_ai_patterns.json": {
        "stage": "deai_cleanup",
        "priority": 1,
        "load_mode": "always",
        "budget_weight": "high",
        "tags": ["anti_ai", "ai_patterns", "cliche", "humanize"],
        "intents": [],
        "block_name": "stylistic_issues",
    },
    "officialese_cliches.json": {
        "stage": "deai_cleanup",
        "priority": 1,
        "load_mode": "always",
        "budget_weight": "high",
        "tags": ["officialese", "cliche", "bureaucratic"],
        "intents": [],
        "block_name": "stylistic_issues",
    },
    "stop_words.json": {
        "stage": "deai_cleanup",
        "priority": 1,
        "load_mode": "always",
        "budget_weight": "high",
        "tags": ["stop_words", "filler", "deai", "word_level"],
        "intents": [],
        # stop_words — отдельный блок, не объединяется
    },

    # ── editorial_core (priority 2) ───────────────────────────────────────
    "general_editing_principles.json": {
        "stage": "editorial_core",
        "priority": 2,
        "load_mode": "always",
        "budget_weight": "high",
        "tags": ["editing", "principles", "style", "clarity"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "types_of_editing.json": {
        "stage": "editorial_core",
        "priority": 2,
        "load_mode": "always",
        "budget_weight": "medium",
        "tags": ["editing", "types", "workflow"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "editorial_analysis_algorithm.json": {
        "stage": "editorial_core",
        "priority": 2,
        "load_mode": "always",
        "budget_weight": "high",
        "tags": ["algorithm", "editing", "analysis", "workflow"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "focus_on_reader_and_goal.json": {
        "stage": "editorial_core",
        "priority": 2,
        "load_mode": "always",
        "budget_weight": "high",
        "tags": ["reader", "goal", "audience", "clarity"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "milchin_techniques.json": {
        "stage": "editorial_core",
        "priority": 2,
        "load_mode": "always",
        "budget_weight": "medium",
        "tags": ["milchin", "techniques", "style", "clarity"],
        "intents": [],
        "block_name": "stylistic_issues",
    },
    "composition_principles.json": {
        "stage": "editorial_core",
        "priority": 2,
        "load_mode": "always",
        "budget_weight": "medium",
        "tags": ["composition", "structure", "principles"],
        "intents": [],
        # отдельный блок, не объединяется
    },
    "local_cohesion.json": {
        "stage": "editorial_core",
        "priority": 2,
        "load_mode": "always",
        "budget_weight": "medium",
        "tags": ["cohesion", "transitions", "flow", "paragraph"],
        "intents": [],
        # отдельный блок
    },

    # ── stylistic_diagnosis (priority 3) ──────────────────────────────────
    "lexical_semantic_errors.json": {
        "stage": "stylistic_diagnosis",
        "priority": 3,
        "load_mode": "always",
        "budget_weight": "medium",
        "tags": ["lexical", "semantic", "errors", "word_choice"],
        "intents": [],
        "block_name": "stylistic_issues",
    },
    "syntax_errors.json": {
        "stage": "stylistic_diagnosis",
        "priority": 3,
        "load_mode": "always",
        "budget_weight": "medium",
        "tags": ["syntax", "errors", "sentence", "structure"],
        "intents": [],
        "block_name": "stylistic_issues",
    },
    "composition_errors.json": {
        "stage": "stylistic_diagnosis",
        "priority": 3,
        "load_mode": "always",
        "budget_weight": "medium",
        "tags": ["composition", "errors", "structure"],
        "intents": [],
        # отдельный блок
    },
    "style_register_errors.json": {
        "stage": "stylistic_diagnosis",
        "priority": 3,
        "load_mode": "by_tags",
        "budget_weight": "medium",
        "tags": ["register", "tone", "style", "mismatch"],
        "intents": [],
        "block_name": "stylistic_issues",
    },

    # ── word_level (priority 4) ────────────────────────────────────────────
    "cleaning_words_and_noise.json": {
        "stage": "word_level",
        "priority": 4,
        "load_mode": "always",
        "budget_weight": "medium",
        "tags": ["word_level", "noise", "filler", "clarity"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "syntax_and_sentence_structure.json": {
        "stage": "word_level",
        "priority": 4,
        "load_mode": "always",
        "budget_weight": "medium",
        "tags": ["syntax", "sentence", "structure", "rhythm"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "tautology_pleonasm.json": {
        "stage": "word_level",
        "priority": 4,
        "load_mode": "always",
        "budget_weight": "low",
        "tags": ["tautology", "pleonasm", "redundancy", "word_level"],
        "intents": [],
        "block_name": "stylistic_issues",
    },
    "nkrj_structure_patterns.json": {
        "stage": "word_level",
        "priority": 4,
        "load_mode": "by_tags",
        "budget_weight": "low",
        "tags": ["nkrj", "corpus", "passive", "sentence_length"],
        "intents": [],
        # отдельный блок
    },

    # ── composition (priority 5) ───────────────────────────────────────────
    "paragraph_structure_and_composition.json": {
        "stage": "composition",
        "priority": 5,
        "load_mode": "always",
        "budget_weight": "medium",
        "tags": ["paragraph", "composition", "structure", "flow"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "compositional_editing.json": {
        "stage": "composition",
        "priority": 5,
        "load_mode": "always",
        "budget_weight": "medium",
        "tags": ["composition", "editing", "macro_structure"],
        "intents": [],
        "block_name": "editorial_techniques",
    },

    # ── grammar_safety (priority 6) ───────────────────────────────────────
    "grammatical_editing.json": {
        "stage": "grammar_safety",
        "priority": 6,
        "load_mode": "always",
        "budget_weight": "low",
        "tags": ["grammar", "morphology", "agreement", "case"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "grammar_errors.json": {
        "stage": "grammar_safety",
        "priority": 6,
        "load_mode": "always",
        "budget_weight": "low",
        "tags": ["grammar", "errors", "correction"],
        "intents": [],
        # отдельный блок
    },
    "grammar_morphology.json": {
        "stage": "grammar_safety",
        "priority": 6,
        "load_mode": "by_tags",
        "budget_weight": "low",
        "tags": ["grammar", "morphology", "forms", "inflection"],
        "intents": [],
        "block_name": "stylistic_issues",
    },

    # ── logic (priority 6, параллельно с grammar) ─────────────────────────
    "logic_issues.json": {
        "stage": "logic",
        "priority": 6,
        "load_mode": "by_tags",
        "budget_weight": "medium",
        "tags": ["logic", "argumentation", "coherence", "reasoning"],
        "intents": [],
        # отдельный блок
    },
    "logical_structure.json": {
        "stage": "logic",
        "priority": 6,
        "load_mode": "by_tags",
        "budget_weight": "medium",
        "tags": ["logic", "structure", "argumentation", "macro_structure"],
        "intents": [],
        "block_name": "editorial_techniques",
    },

    # ── specialized (priority 7, по тегам) ────────────────────────────────
    "foreign_words_and_translation.json": {
        "stage": "specialized",
        "priority": 7,
        "load_mode": "by_tags",
        "budget_weight": "low",
        "tags": ["foreign_words", "translation", "borrowings", "terminology"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "ethics_and_tact.json": {
        "stage": "specialized",
        "priority": 7,
        "load_mode": "by_tags",
        "budget_weight": "low",
        "tags": ["ethics", "tact", "sensitivity", "audience"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "cultural_and_factual_checks.json": {
        "stage": "specialized",
        "priority": 7,
        "load_mode": "by_tags",
        "budget_weight": "low",
        "tags": ["cultural", "factual", "accuracy", "verification"],
        "intents": [],
        "block_name": "editorial_techniques",
    },
    "phonetics_word_formation.json": {
        "stage": "specialized",
        "priority": 7,
        "load_mode": "by_tags",
        "budget_weight": "low",
        "tags": ["phonetics", "word_formation", "euphony", "sound"],
        "intents": [],
        "block_name": "stylistic_issues",
    },

    # ── rhetoric (priority 8, по интенту) ─────────────────────────────────
    "rhetoric_topoi.json": {
        "stage": "rhetoric",
        "priority": 8,
        "load_mode": "by_intent",
        "budget_weight": "low",
        "tags": ["rhetoric", "topoi", "argumentation", "persuasion"],
        "intents": ["argumentation", "persuasion", "rhetoric", "academic"],
        "block_name": "rhetoric_frameworks",
    },
    "rhetoric_figures.json": {
        "stage": "rhetoric",
        "priority": 8,
        "load_mode": "by_intent",
        "budget_weight": "low",
        "tags": ["rhetoric", "figures", "style", "expressiveness"],
        "intents": ["rhetoric", "expressiveness", "literary", "speech"],
        "block_name": "rhetoric_frameworks",
    },
    "rhetoric_tropes_and_strategies.json": {
        "stage": "rhetoric",
        "priority": 8,
        "load_mode": "by_intent",
        "budget_weight": "low",
        "tags": ["rhetoric", "tropes", "strategies", "persuasion"],
        "intents": ["rhetoric", "persuasion", "argumentation"],
        "block_name": "rhetoric_frameworks",
    },
    "phraseology_tropes.json": {
        "stage": "rhetoric",
        "priority": 8,
        "load_mode": "by_intent",
        "budget_weight": "low",
        "tags": ["phraseology", "tropes", "idioms", "expressiveness"],
        "intents": ["rhetoric", "expressiveness", "literary"],
        "block_name": "stylistic_issues",
    },
    "imagery_and_style.json": {
        "stage": "rhetoric",
        "priority": 8,
        "load_mode": "by_intent",
        "budget_weight": "low",
        "tags": ["imagery", "metaphor", "style", "expressiveness"],
        "intents": ["rhetoric", "expressiveness", "literary", "creative"],
        "block_name": "editorial_techniques",
    },

    # ── overlays: storytelling (по интенту) ───────────────────────────────
    "storytelling_macrostructures.json": {
        "stage": "overlay_storytelling",
        "priority": 9,
        "load_mode": "by_intent",
        "budget_weight": "medium",
        "tags": ["storytelling", "narrative", "structure", "macro_structure"],
        "intents": ["storytelling", "narrative", "blog", "creative"],
        "block_name": "storytelling_frameworks",
    },
    "storytelling_microtechniques.json": {
        "stage": "overlay_storytelling",
        "priority": 9,
        "load_mode": "by_intent",
        "budget_weight": "medium",
        "tags": ["storytelling", "micro_technique", "show_not_tell", "suspense"],
        "intents": ["storytelling", "narrative", "creative"],
        "block_name": "storytelling_frameworks",
    },

    # ── overlays: marketing (по интенту) ──────────────────────────────────
    "marketing_web.json": {
        "stage": "overlay_marketing",
        "priority": 9,
        "load_mode": "by_intent",
        "budget_weight": "medium",
        "tags": ["marketing", "web", "landing", "cta"],
        "intents": ["marketing", "landing_page", "product"],
        "block_name": "marketing_templates",
    },
    "marketing_email.json": {
        "stage": "overlay_marketing",
        "priority": 9,
        "load_mode": "by_intent",
        "budget_weight": "medium",
        "tags": ["marketing", "email", "newsletter", "lead_magnet"],
        "intents": ["marketing", "email"],
        "block_name": "marketing_templates",
    },
    "marketing_social.json": {
        "stage": "overlay_marketing",
        "priority": 9,
        "load_mode": "by_intent",
        "budget_weight": "medium",
        "tags": ["marketing", "social", "smm", "post"],
        "intents": ["marketing", "social_media"],
        "block_name": "marketing_templates",
    },
    "marketing_other.json": {
        "stage": "overlay_marketing",
        "priority": 9,
        "load_mode": "by_intent",
        "budget_weight": "low",
        "tags": ["marketing", "case_study", "presentation"],
        "intents": ["marketing"],
        "block_name": "marketing_templates",
    },

    # ── overlays: genre / final check (по интенту) ────────────────────────
    "genre_templates.json": {
        "stage": "overlay_genre",
        "priority": 9,
        "load_mode": "by_intent",
        "budget_weight": "low",
        "tags": ["genre", "template", "format", "structure"],
        "intents": ["genre", "academic", "journalistic", "business"],
        "block_name": "editorial_techniques",
    },
    "final_check_cta_readability.json": {
        "stage": "overlay_final",
        "priority": 9,
        "load_mode": "by_intent",
        "budget_weight": "low",
        "tags": ["final_check", "cta", "readability", "proofreading"],
        "intents": ["final_check", "proofreading", "marketing"],
        "block_name": "editorial_techniques",
    },
}

# ── Подпапки, в которых лежат файлы ──────────────────────────────────────────
# Скрипт ищет файлы рекурсивно, но запоминает подпапку как "subfolder" в записи.
KNOWN_SUBFOLDERS = ("editorial_techniques", "stylistic_issues")


# ── Основная логика ───────────────────────────────────────────────────────────

def collect_kb_files(kb_dir: Path) -> list[tuple[Path, str]]:
    """Возвращает список (путь_к_файлу, относительный_путь_от_kb_dir)."""
    result: list[tuple[Path, str]] = []
    for json_file in sorted(kb_dir.rglob("*.json")):
        if json_file.name == MANIFEST_FILENAME:
            continue
        rel = json_file.relative_to(kb_dir)
        result.append((json_file, str(rel).replace("\\", "/")))
    return result


def build_entry(file_path: Path, rel_path: str, kb_dir: Path) -> dict[str, Any]:
    """Строит одну запись манифеста для файла."""
    name = file_path.name
    rule = FILE_RULES.get(name)

    parts = Path(rel_path).parts
    subfolder = parts[0] if len(parts) > 1 else ""

    # Размер файла в KB
    size_kb = round(file_path.stat().st_size / 1024, 1)

    if rule:
        entry = {
            "file": rel_path,
            "subfolder": subfolder,
            "size_kb": size_kb,
            "stage": rule["stage"],
            "priority": rule["priority"],
            "load_mode": rule["load_mode"],
            "budget_weight": rule["budget_weight"],
            "tags": rule["tags"],
            "intents": rule["intents"],
            "status": "active",
            "note": "",
        }
        # Добавляем block_name, если он указан
        if "block_name" in rule:
            entry["block_name"] = rule["block_name"]
    else:
        # Файл не описан в правилах — помечаем как unclassified
        entry = {
            "file": rel_path,
            "subfolder": subfolder,
            "size_kb": size_kb,
            "stage": "unclassified",
            "priority": 99,
            "load_mode": "never",
            "budget_weight": "low",
            "tags": [],
            "intents": [],
            "status": "unclassified",
            "note": "Файл не найден в FILE_RULES — проверь и добавь вручную",
        }
    return entry


def generate_manifest(kb_dir: Path, dry_run: bool, update: bool) -> None:
    out_path = kb_dir / MANIFEST_FILENAME

    # При --update читаем существующий манифест, чтобы сохранить ручные заметки
    existing: dict[str, dict] = {}
    if update and out_path.exists():
        old = json.loads(out_path.read_text(encoding="utf-8"))
        for e in old.get("files", []):
            existing[e["file"]] = e
        print(f"  ℹ️  Режим --update: загружено {len(existing)} существующих записей")

    files = collect_kb_files(kb_dir)
    entries: list[dict] = []
    unclassified: list[str] = []

    for file_path, rel_path in files:
        entry = build_entry(file_path, rel_path, kb_dir)

        # При --update: перенести ручные заметки из старого манифеста
        if update and rel_path in existing:
            old_entry = existing[rel_path]
            if old_entry.get("note"):
                entry["note"] = old_entry["note"]

        entries.append(entry)
        if entry["status"] == "unclassified":
            unclassified.append(rel_path)

    # Сортируем: сначала по priority, потом по file
    entries.sort(key=lambda e: (e["priority"], e["file"]))

    # Статистика по слоям
    stages: dict[str, int] = {}
    for e in entries:
        stages[e["stage"]] = stages.get(e["stage"], 0) + 1

    manifest: dict[str, Any] = {
        "_meta": {
            "description": "Манифест базы знаний. Управляет загрузкой файлов в prompt_builder.",
            "generated_by": "generate_kb_manifest.py",
            "total_files": len(entries),
            "stages": stages,
            "load_modes": {
                "always": "Файл грузится при каждом запросе",
                "by_tags": "Файл грузится, если теги текста совпадают с tags[]",
                "by_intent": "Файл грузится только при явном интенте из intents[]",
                "never": "Файл отключён (unclassified или вручную)",
            },
            "budget_weights": {
                "high":   "Урезается последним при сжатии промпта",
                "medium": "Урезается вторым",
                "low":    "Урезается первым",
            },
        },
        "files": entries,
    }

    pretty = json.dumps(manifest, ensure_ascii=False, indent=2)
    size_kb = len(pretty.encode("utf-8")) / 1024

    print(f"\n📋  Манифест: {len(entries)} файлов, {size_kb:.1f} KB")
    print(f"    Слои: {stages}")

    if unclassified:
        print(f"\n⚠️   Не классифицировано ({len(unclassified)}):")
        for f in unclassified:
            print(f"      · {f}")

    if dry_run:
        print("\n[dry-run] Файл не записан.")
    else:
        out_path.write_text(pretty, encoding="utf-8")
        print(f"\n✅  Записано: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генерирует kb_manifest.json по реальной структуре knowledge_base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kb-dir", type=Path, default=None,
        help="Путь к папке knowledge_base (по умолчанию: папка рядом со скриптом).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Показать результат без записи файла.",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Обновить существующий манифест, сохранив ручные заметки (note).",
    )
    args = parser.parse_args()

    kb_dir = (args.kb_dir or Path(__file__).parent / "knowledge_base").resolve()
    if not kb_dir.exists():
        print(f"✗ Папка не найдена: {kb_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"generate_kb_manifest  |  KB: {kb_dir}")
    generate_manifest(kb_dir, dry_run=args.dry_run, update=args.update)


if __name__ == "__main__":
    main()