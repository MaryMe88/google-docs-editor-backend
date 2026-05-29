from __future__ import annotations

from typing import Final, Set

ALLOWED_DOMAINS: Final[Set[str]] = {"marketing", "blog", "deai"}

ALLOWED_INTENTS: Final[Set[str]] = {
    "storytelling",
    "noragal",
    "deai",
    "neutral",
}

# Синхронизировано с реальными файлами config/overlays/*.json
# Фактически присутствующие на сервере:
#   coldemail, factcheck, finalcheck, infostyle, landing,
#   pressrelease, readerfocus, recommendations, structurefirst, workdoc
ALLOWED_OVERLAYS: Final[Set[str]] = {
    "coldemail",
    "factcheck",
    "finalcheck",
    "infostyle",
    "landing",
    "pressrelease",
    "readerfocus",
    "recommendations",
    "structurefirst",
    "workdoc",
}

ALLOWED_OUTPUT_MODES: Final[Set[str]] = {"text_only", "text_and_report"}

ALLOWED_PROVIDERS: Final[Set[str]] = {
    "openrouter",
    "perplexity",
    "openai",
    "anthropic",
}

ALLOWED_KIND: Final[Set[str]] = {"b2b", "b2c", "mixed", "custom"}
ALLOWED_EXPERTISE: Final[Set[str]] = {"novice", "pro", "expert"}
ALLOWED_FORMALITY: Final[Set[str]] = {"casual", "neutral", "formal"}
