"""
config_models.cache

Cache-инфраструктура: CachePolicy, _CacheEntry, FileCache.

Выделено из src.config_types в итерации 7 дорожной карты.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Generic,
    TypeVar,
)

V = TypeVar("V")


@dataclass
class CachePolicy:
    """
    Политика инвалидации кэша.

    Атрибуты:
        check_mtime: Инвалидировать при изменении mtime файла.
        ttl_seconds: Время жизни кэша в секундах (None = без TTL).

    Рекомендуемые режимы:
        prod: CachePolicy(check_mtime=True)
        dev: CachePolicy(check_mtime=True, ttl_seconds=30)
        test: CachePolicy(check_mtime=False, ttl_seconds=None)
    """

    check_mtime: bool = True
    ttl_seconds: float | None = None


@dataclass
class _CacheEntry(Generic[V]):
    """Внутренняя запись кэша."""

    value: V
    path: Path | None
    loaded_at: float
    mtime_at_load: float | None


class FileCache:
    """
    Кэш файловых данных с поддержкой TTL и mtime-инвалидации.

    Использование:
        cache = FileCache(policy=CachePolicy(check_mtime=True))
        data = cache.get_or_load("key", path, loader_fn, *loader_args)
    """

    def __init__(self, policy: CachePolicy | None = None) -> None:
        self._policy = policy or CachePolicy(check_mtime=True)
        self._store: dict[str, _CacheEntry[Any]] = {}

    def _is_valid(self, entry: _CacheEntry[Any]) -> bool:
        """Проверяет актуальность записи кэша."""
        now = time.monotonic()

        if self._policy.ttl_seconds is not None:
            if now - entry.loaded_at > self._policy.ttl_seconds:
                return False

        if self._policy.check_mtime and entry.path is not None:
            try:
                current_mtime = entry.path.stat().st_mtime
                if entry.mtime_at_load is None or current_mtime != entry.mtime_at_load:
                    return False
            except OSError:
                return False

        return True

    def get_or_load(
        self,
        key: str,
        path: Path | None,
        loader: Callable[..., V],
        *loader_args: Any,
    ) -> V:
        """
        Возвращает закэшированное значение или загружает через loader(*loader_args).

        Args:
            key: Ключ кэша (уникальный идентификатор значения).
            path: Путь к файлу для mtime-инвалидации (None = без mtime).
            loader: Callable, возвращающий значение.
            loader_args: Позиционные аргументы для loader.
        """
        entry = self._store.get(key)
        if entry is not None and self._is_valid(entry):
            return entry.value

        value = loader(*loader_args)

        mtime: float | None = None
        if path is not None and self._policy.check_mtime:
            with suppress(OSError):
                mtime = path.stat().st_mtime

        self._store[key] = _CacheEntry(
            value=value,
            path=path,
            loaded_at=time.monotonic(),
            mtime_at_load=mtime,
        )
        return value

    def get_or_load_multi(
        self,
        key: str,
        paths: list[Path],
        loader: Callable[..., V],
        *loader_args: Any,
    ) -> V:
        """
        Кэширует результат loader с инвалидацией по нескольким файлам.
        Инвалидируется если изменился mtime любого из paths.
        """
        entry = self._store.get(key)

        if entry is not None:
            if self._policy.ttl_seconds is not None:
                if time.monotonic() - entry.loaded_at > self._policy.ttl_seconds:
                    entry = None

            if entry is not None and self._policy.check_mtime and entry.mtime_at_load is not None:
                try:
                    current_max = max(
                        (path.stat().st_mtime for path in paths if path.exists()),
                        default=0.0,
                    )
                    if current_max != entry.mtime_at_load:
                        entry = None
                except OSError:
                    entry = None

            if entry is not None:
                return entry.value

        value = loader(*loader_args)

        try:
            max_mtime: float | None = max(
                (path.stat().st_mtime for path in paths if path.exists()),
                default=None,
            )
        except OSError:
            max_mtime = None

        self._store[key] = _CacheEntry(
            value=value,
            path=None,
            loaded_at=time.monotonic(),
            mtime_at_load=max_mtime,
        )
        return value

    def invalidate(self, key: str) -> None:
        """Удаляет запись из кэша."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Сбрасывает весь кэш."""
        self._store.clear()
