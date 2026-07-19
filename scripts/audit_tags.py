from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


ROOT = Path(__file__).resolve().parent.parent
TAG_MAP_PATH = ROOT / "config" / "tag_map.json"
KB_PATH = ROOT / "knowledge_base"


def normalize_tag(tag: str) -> str:
    return tag.strip().lower()


def compact_tag(tag: str) -> str:
    return normalize_tag(tag).replace("_", "").replace("-", "").replace(" ", "")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_tags_from_mapping(node: Any, acc: Set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"primary", "expanded"} and isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        acc.add(normalize_tag(item))
            else:
                extract_tags_from_mapping(value, acc)
    elif isinstance(node, list):
        for item in node:
            extract_tags_from_mapping(item, acc)


def iter_all_values(node: Any) -> Iterable[Any]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from iter_all_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_all_values(item)


def extract_tags_from_kb_file(data: Any) -> Set[str]:
    tags: Set[str] = set()

    for node in iter_all_values(data):
        if not isinstance(node, dict):
            continue

        raw_tags = node.get("tags")
        if isinstance(raw_tags, list):
            for item in raw_tags:
                if isinstance(item, str) and item.strip():
                    tags.add(normalize_tag(item))
        elif isinstance(raw_tags, str) and raw_tags.strip():
            tags.add(normalize_tag(raw_tags))

    return tags


def load_tag_map_tags(tag_map_path: Path) -> Set[str]:
    data = load_json(tag_map_path)
    tags: Set[str] = set()
    extract_tags_from_mapping(data, tags)
    return tags


def load_kb_tags(kb_path: Path) -> Tuple[Set[str], Dict[str, List[str]]]:
    all_tags: Set[str] = set()
    tag_sources: Dict[str, List[str]] = {}

    for json_file in sorted(kb_path.rglob("*.json")):
        try:
            data = load_json(json_file)
        except Exception as exc:
            print(f"⚠️ Failed to read {json_file}: {exc}")
            continue

        file_tags = extract_tags_from_kb_file(data)
        for tag in file_tags:
            all_tags.add(tag)
            tag_sources.setdefault(tag, []).append(str(json_file.relative_to(ROOT)))

    return all_tags, tag_sources


def try_load_canonical_tags() -> Set[str]:
    try:
        sys.path.insert(0, str(ROOT))
        from src.startup_checks import CANONICAL_TAGS  # type: ignore
    except Exception as exc:
        print(f"⚠️ Could not import CANONICAL_TAGS: {exc}")
        return set()

    result: Set[str] = set()
    for tag in CANONICAL_TAGS:
        if isinstance(tag, str) and tag.strip():
            result.add(normalize_tag(tag))
    return result


def find_naming_collisions(tags: Set[str]) -> List[Tuple[str, List[str]]]:
    groups: Dict[str, List[str]] = {}
    for tag in sorted(tags):
        groups.setdefault(compact_tag(tag), []).append(tag)

    collisions: List[Tuple[str, List[str]]] = []
    for compact, variants in groups.items():
        if len(variants) > 1:
            collisions.append((compact, variants))
    return collisions


def print_section(title: str, items: Iterable[str]) -> None:
    items = sorted(set(items))
    print(f"\n{title}")
    print("-" * len(title))
    if not items:
        print("None")
        return
    for item in items:
        print(f"- {item}")


def main() -> None:
    print("Tag audit")
    print("=========")
    print(f"ROOT     : {ROOT}")
    print(f"tag_map  : {TAG_MAP_PATH}")
    print(f"kb       : {KB_PATH}")

    if not TAG_MAP_PATH.exists():
        raise FileNotFoundError(f"tag_map.json not found: {TAG_MAP_PATH}")
    if not KB_PATH.exists():
        raise FileNotFoundError(f"knowledge_base directory not found: {KB_PATH}")

    tag_map_tags = load_tag_map_tags(TAG_MAP_PATH)
    kb_tags, tag_sources = load_kb_tags(KB_PATH)
    canonical_tags = try_load_canonical_tags()

    print(f"\nTag counts")
    print("----------")
    print(f"tag_map tags    : {len(tag_map_tags)}")
    print(f"kb tags         : {len(kb_tags)}")
    print(f"canonical tags  : {len(canonical_tags)}")

    missing_in_kb = tag_map_tags - kb_tags
    extra_in_kb = kb_tags - tag_map_tags

    print_section("Tags in tag_map but missing in KB", missing_in_kb)
    print_section("Tags in KB but missing in tag_map", extra_in_kb)

    if canonical_tags:
        print_section(
            "Tags in CANONICAL_TAGS but missing in KB",
            canonical_tags - kb_tags,
        )
        print_section(
            "Tags in CANONICAL_TAGS but missing in tag_map",
            canonical_tags - tag_map_tags,
        )
        print_section(
            "Tags in tag_map but missing in CANONICAL_TAGS",
            tag_map_tags - canonical_tags,
        )

    combined_tags = tag_map_tags | kb_tags | canonical_tags
    collisions = find_naming_collisions(combined_tags)

    print("\nPotential naming collisions")
    print("---------------------------")
    if not collisions:
        print("None")
    else:
        for compact, variants in collisions:
            print(f"- {compact}: {', '.join(variants)}")

    print("\nSample KB sources for tags missing in tag_map")
    print("---------------------------------------------")
    if not extra_in_kb:
        print("None")
    else:
        for tag in sorted(extra_in_kb):
            sources = tag_sources.get(tag, [])
            sample = ", ".join(sources[:3])
            print(f"- {tag}: {sample}")


if __name__ == "__main__":
    main()