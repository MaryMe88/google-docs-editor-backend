"""
services.provider_health

Кэш доступности LLM-провайдеров и проверка их доступности
(env-based light check и deep HTTP check).

Выделено из src.main в итерации 8 дорожной карты.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from src.llm_client import create_llm_client
from src.provider_registry import LLMProvider
from src.shared_contracts import ALLOWED_PROVIDERS

logger = logging.getLogger(__name__)


@dataclass
class _ProviderCacheEntry:
    available: bool
    checked_at: float = field(default_factory=time.monotonic)

    def is_fresh(self, ttl: float) -> bool:
        return (time.monotonic() - self.checked_at) < ttl


_provider_cache: dict[str, _ProviderCacheEntry] = {}
_PROVIDER_CACHE_TTL = 60.0


def _get_cached_availability(provider: str) -> bool | None:
    entry = _provider_cache.get(provider)
    if entry and entry.is_fresh(_PROVIDER_CACHE_TTL):
        return entry.available
    return None


def _set_cached_availability(provider: str, available: bool) -> None:
    _provider_cache[provider] = _ProviderCacheEntry(available=available)


def invalidate_provider_cache() -> None:
    _provider_cache.clear()


_PROVIDER_KEY_ENV: dict[str, str] = {
    "perplexity": "PERPLEXITY_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


async def _check_provider_deep(provider_name: str) -> bool:
    try:
        provider_enum = LLMProvider(provider_name)
        async with create_llm_client(
            provider=provider_enum,
            model=None,
            temperature=0.0,
            timeout=5.0,
            max_retries=1,
            max_tokens=1,
        ) as client:
            await client.generate("ping")
            return True
    except Exception as error:
        logger.debug("Deep check failed for %s: %s", provider_name, error)
        return False


async def _check_providers_availability(
    deep: bool = False,
) -> tuple[bool, dict[str, bool]]:
    results: dict[str, bool] = {}
    for provider in ALLOWED_PROVIDERS:
        if not deep:
            cached = _get_cached_availability(provider)
            if cached is not None:
                results[provider] = cached
                continue
            env_var = _PROVIDER_KEY_ENV.get(provider.lower())
            available = bool(os.getenv(env_var)) if env_var else False
            _set_cached_availability(provider, available)
            results[provider] = available
        else:
            available = await _check_provider_deep(provider)
            _set_cached_availability(provider, available)
            results[provider] = available

    any_available = any(results.values())
    return any_available, results
