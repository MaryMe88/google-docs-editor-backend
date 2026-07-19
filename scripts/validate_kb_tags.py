from __future__ import annotations

import sys
from pathlib import Path


"""Validate that every tag declared in CANONICAL_TAGS exists in at least one KB record.

Usage (local):
    python scripts/validate_kb_tags.py

Also runs automatically as a CI step before deploy.
Exit code 0 = OK, exit code 1 = validation failed.
"""

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from src.shared_contracts import (
        ALLOWED_DOMAINS,
        ALLOWED_INTENTS,
        ALLOWED_OVERLAYS,
    )
    from src.startup_checks import run_startup_checks
except ImportError as exc:
    print(f"❌ Import error — make sure dependencies are installed: {exc}")
    sys.exit(1)


def main() -> None:
    config_path = ROOT / "config"
    kb_path = ROOT / "knowledge_base"

    print("Checking KB tags...")
    print(f" config : {config_path}")
    print(f" kb     : {kb_path}")
    print()

    try:
        run_startup_checks(
            allowed_domains=ALLOWED_DOMAINS,
            allowed_intents=ALLOWED_INTENTS,
            allowed_overlays=ALLOWED_OVERLAYS,
            config_path=config_path,
            kb_path=kb_path,
        )
    except RuntimeError as exc:
        print(f"❌ KB validation failed:\n\n{exc}")
        print(
            "\nFix: add the missing tag(s) to at least one record's 'tags' field "
            "in knowledge_base/*.json"
        )
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"❌ Missing file or directory: {exc}")
        sys.exit(1)

    print("✅ KB validation passed — all CANONICAL_TAGS are present in the knowledge base.")


if __name__ == "__main__":
    main()