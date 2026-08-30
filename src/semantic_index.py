"""
semantic_index.py

Семантический индекс для поиска правил базы знаний по смыслу.
Использует sentence-transformers (rubert-tiny2) и numpy для косинусного поиска.
Embeddings кешируются в файл kb_embeddings.npy и пересчитываются,
если JSON-файлы базы знаний изменились.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Путь к кешу embeddings рядом с папкой knowledge_base
_CACHE_PATH = Path(__file__).parent.parent / "knowledge_base" / "kb_embeddings.npy"
_CACHE_META_PATH = Path(__file__).parent.parent / "knowledge_base" / "kb_embeddings_meta.json"


class SemanticIndex:
    """
    Индекс для семантического поиска записей базы знаний.

    Пример использования:
        index = SemanticIndex()
        index.build(all_entries)          # один раз при старте
        results = index.search(query, top_k=10)
    """

    def __init__(self, model_name: str = "cointegrated/rubert-tiny2") -> None:
        self.model_name = model_name
        self._model = None          # ленивая загрузка
        self.entries: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self._is_built = False

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def build(self, all_entries: list[dict[str, Any]], force_rebuild: bool = False) -> None:
        """
        Строит индекс из списка записей базы знаний.
        Если кеш актуален — загружает из файла, иначе пересчитывает.

        :param all_entries: все записи из всех JSON-файлов knowledge_base
        :param force_rebuild: пересчитать embeddings даже если кеш есть
        """
        self.entries = all_entries

        if not force_rebuild and self._load_cache(len(all_entries)):
            logger.info(
                "SemanticIndex: загружен кеш embeddings (%d записей)", len(all_entries)
            )
            self._is_built = True
            return

        logger.info(
            "SemanticIndex: строю embeddings для %d записей (модель: %s)…",
            len(all_entries),
            self.model_name,
        )
        t0 = time.monotonic()
        texts = [self._entry_to_text(e) for e in all_entries]
        model = self._get_model()
        self.embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        elapsed = time.monotonic() - t0
        logger.info("SemanticIndex: embeddings готовы за %.1f сек", elapsed)

        self._save_cache(len(all_entries))
        self._is_built = True

    def search(
        self, query: str, top_k: int = 10
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Ищет top_k наиболее семантически близких записей.

        :returns: список пар (entry, score), отсортированных по убыванию score
        """
        if not self._is_built or self.embeddings is None:
            logger.warning("SemanticIndex.search вызван до build() — возвращаю пустой список")
            return []

        if not query or not query.strip():
            return []

        model = self._get_model()
        query_emb = model.encode(
            [query.strip()],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores: np.ndarray = (self.embeddings @ query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.entries[i], float(scores[i])) for i in top_indices]

    def is_ready(self) -> bool:
        """Возвращает True, если индекс построен и готов к поиску."""
        return self._is_built and self.embeddings is not None

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _get_model(self):
        """Ленивая загрузка модели — загружается один раз при первом обращении."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("SemanticIndex: загружаю модель %s…", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                logger.info("SemanticIndex: модель загружена")
            except ImportError:
                raise ImportError(
                    "Библиотека sentence-transformers не установлена. "
                    "Добавь её в requirements.txt: sentence-transformers>=3.0.0"
                )
        return self._model

    @staticmethod
    def _entry_to_text(entry: dict[str, Any]) -> str:
        """
        Превращает запись базы знаний в одну строку для индексации.
        Берёт самые информативные поля.
        """
        fields = ("name", "description", "rule", "wrong", "when_to_use")
        parts = []
        for field in fields:
            value = entry.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
            elif isinstance(value, list):
                parts.extend(v for v in value if isinstance(v, str) and v.strip())
        return " ".join(parts)

    def _load_cache(self, expected_count: int) -> bool:
        """Загружает кеш embeddings, если он актуален. Возвращает True при успехе."""
        try:
            if not _CACHE_PATH.exists() or not _CACHE_META_PATH.exists():
                return False

            with _CACHE_META_PATH.open(encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("count") != expected_count:
                logger.debug(
                    "SemanticIndex: кеш устарел (count %s != %s)",
                    meta.get("count"), expected_count,
                )
                return False

            if meta.get("model") != self.model_name:
                logger.debug("SemanticIndex: кеш от другой модели, пересчитываю")
                return False

            # SEC-патч 4.1: явный allow_pickle=False
            self.embeddings = np.load(str(_CACHE_PATH), allow_pickle=False)
            return True

        except Exception as exc:
            logger.warning("SemanticIndex: не удалось загрузить кеш: %s", exc)
            return False

    def _save_cache(self, count: int) -> None:
        """Сохраняет embeddings и метаданные на диск."""
        try:
            np.save(str(_CACHE_PATH), self.embeddings)
            meta = {"count": count, "model": self.model_name, "built_at": time.time()}
            with _CACHE_META_PATH.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logger.info("SemanticIndex: кеш сохранён (%s)", _CACHE_PATH)
        except Exception as exc:
            logger.warning("SemanticIndex: не удалось сохранить кеш: %s", exc)


# ------------------------------------------------------------------
# Глобальный экземпляр (singleton) — инициализируется в main.py
# ------------------------------------------------------------------
_global_index: SemanticIndex | None = None


def get_semantic_index() -> SemanticIndex | None:
    """Возвращает глобальный индекс, если он инициализирован."""
    return _global_index


def init_semantic_index(
    all_entries: list[dict[str, Any]],
    model_name: str = "cointegrated/rubert-tiny2",
    force_rebuild: bool = False,
) -> SemanticIndex:
    """
    Инициализирует и строит глобальный семантический индекс.
    Вызывать один раз при старте приложения (например, в lifespan FastAPI).

    Пример в main.py:
        from src.semantic_index import init_semantic_index
        all_entries = kb.grammar_errors + kb.stylistic_issues + kb.logic_issues
        init_semantic_index(all_entries)
    """
    global _global_index
    index = SemanticIndex(model_name=model_name)
    index.build(all_entries, force_rebuild=force_rebuild)
    _global_index = index
    logger.info("SemanticIndex: глобальный индекс готов (%d записей)", len(all_entries))
    return index