from __future__ import annotations

from typing import Final, Set

# ---------------------------------------------------------------------------
# ДОМЕНЫ
#
# Синхронизировано с реальными файлами config/domains/*.json.
# Каждое значение здесь ДОЛЖНО иметь соответствующий файл
# config/domains/<значение>.json, иначе /api/edit вернёт 500 при сборке промпта.
#
# Фактически присутствующие файлы доменов:
#   basic_edit, blog, composition, cutnoise, deai, fiction, genre,
#   logic_edit, makeclear, marketing, nora_gal, nora_gal_soft,
#   readerfirst, restructure
# ---------------------------------------------------------------------------
ALLOWED_DOMAINS: Final[Set[str]] = {
    "marketing",
    "blog",
    "deai",
    "basic_edit",
    "logic_edit",
    "nora_gal",
    "nora_gal_soft",
    "cutnoise",
    "makeclear",
    "restructure",
    "readerfirst",
    "genre",
    "fiction",
    "composition",
}

# ---------------------------------------------------------------------------
# INTENTS
#
# ВАЖНО: intent пропускается контрактом, ТОЛЬКО если для него есть файл
# config/intents/<нормализованное_значение>.json. Иначе load_intent_config
# в prompt_builder.py упадёт с FileNotFoundError -> HTTP 500.
#
# Здесь оставлены только intent, проверенно работающие на сервере (HTTP 200):
#   - storytelling : есть файл config/intents/storytelling.json
#   - neutral      : специальное значение, prompt_builder трактует его как
#                    "без intent" (load_intent_config возвращает None),
#                    файл не требуется
#
# НЕ добавляй сюда intent без создания соответствующего config/intents/*.json.
# Доменная специфика (nora_gal, deai, basic_edit и т.д.) выражается через
# поле domain, а не через intent.
# ---------------------------------------------------------------------------
ALLOWED_INTENTS: Final[Set[str]] = {
    "storytelling",
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
