from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
TAG_MAP_PATH = ROOT / "config" / "tag_map.json"
BACKUP_PATH = ROOT / "config" / "tag_map.backup.json"

# Только безопасные, очевидные замены по результатам аудита.
RENAMES: Dict[str, str] = {
    "aipatterns": "ai_patterns",
    "antiai": "anti_ai",
    "casestudy": "case_study",
    "discoursemarkers": "discourse_markers",
    "foreignwords": "foreign_words",
    "leadmagnet": "lead_magnet",
    "leftbranching": "left_branching",
    "macrostructure": "macro_structure",
    "microtechnique": "micro_technique",
    "noragal": "nora_gal",
    "passivevoice": "passive_voice",
    "pressrelease": "press_release",
    "sentencelength": "sentence_length",
    "shownottell": "show_not_tell",
    "stopwords": "stop_words",
    "wordchoice": "word_choice",
    "wordformation": "word_formation",
    "wordlevel": "word_level",
}

TARGET_KEYS = {"primary", "expanded"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def rewrite_node(node: Any, path: str = "") -> List[Tuple[str, str, str]]:
    changes: List[Tuple[str, str, str]] = []

    if isinstance(node, dict):
        for key, value in node.items():
            current_path = f"{path}.{key}" if path else key

            if key in TARGET_KEYS and isinstance(value, list):
                new_values: List[str] = []
                local_changes: List[Tuple[str, str, str]] = []

                for item in value:
                    if isinstance(item, str):
                        new_item = RENAMES.get(item, item)
                        new_values.append(new_item)
                        if new_item != item:
                            local_changes.append((current_path, item, new_item))
                    else:
                        new_values.append(item)

                node[key] = dedupe_preserve_order(new_values)
                changes.extend(local_changes)
            else:
                changes.extend(rewrite_node(value, current_path))

    elif isinstance(node, list):
        for index, item in enumerate(node):
            current_path = f"{path}[{index}]"
            changes.extend(rewrite_node(item, current_path))

    return changes


def main() -> None:
    if not TAG_MAP_PATH.exists():
        raise FileNotFoundError(f"tag_map.json not found: {TAG_MAP_PATH}")

    data = load_json(TAG_MAP_PATH)

    shutil.copy2(TAG_MAP_PATH, BACKUP_PATH)

    changes = rewrite_node(data)

    if not changes:
        print("No changes needed.")
        print(f"Backup still created: {BACKUP_PATH}")
        return

    save_json(TAG_MAP_PATH, data)

    print("Updated tag_map.json")
    print(f"Backup saved to: {BACKUP_PATH}")
    print()
    print("Applied changes:")
    for location, old, new in changes:
        print(f"- {location}: {old} -> {new}")
    print()
    print(f"Total replacements: {len(changes)}")


if __name__ == "__main__":
    main()