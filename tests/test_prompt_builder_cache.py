# tests/test_prompt_builder_cache.py
"""
Тесты для кэширования и перезагрузки конфигов в PromptBuilder (четвёртая итерация).

Проверяют:
- get_domain_config, get_intent_config, get_overlay_config, get_output_format кэшируют результаты.
- reload_configs() сбрасывает кэши и данные перечитываются.
- get_knowledge_base использует внутренний кэш (_kb_cache).
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.prompt_builder import PromptBuilder, load_domain_config, load_intent_config, load_overlay_config, load_output_format, load_knowledge_base
from src.config_types import LimitsConfig, KnowledgeBase


@pytest.fixture
def builder():
    return PromptBuilder(config_path=Path("config"), kb_path=Path("knowledge_base"))


class TestPromptBuilderCache:
    """Тесты кэширования конфигов."""

    def test_get_domain_config_caches(self, builder):
        """Повторный вызов get_domain_config должен использовать кэш, не загружая файл."""
        with patch('src.prompt_builder.load_domain_config', wraps=load_domain_config) as mock_load:
            config1 = builder.get_domain_config("blog")
            config2 = builder.get_domain_config("blog")
            assert config1 is config2
            assert mock_load.call_count == 1

    def test_get_intent_config_caches(self, builder):
        with patch('src.prompt_builder.load_intent_config', wraps=load_intent_config) as mock_load:
            config1 = builder.get_intent_config("storytelling")
            config2 = builder.get_intent_config("storytelling")
            assert config1 is config2
            assert mock_load.call_count == 1

    def test_get_overlay_config_caches(self, builder):
        with patch('src.prompt_builder.load_overlay_config', wraps=load_overlay_config) as mock_load:
            config1 = builder.get_overlay_config("infostyle")
            config2 = builder.get_overlay_config("infostyle")
            assert config1 is config2
            assert mock_load.call_count == 1

    def test_get_output_format_caches(self, builder):
        with patch('src.prompt_builder.load_output_format', wraps=load_output_format) as mock_load:
            fmt1 = builder.get_output_format("text_only")
            fmt2 = builder.get_output_format("text_only")
            assert fmt1 == fmt2
            assert mock_load.call_count == 1

    def test_reload_configs_clears_caches(self, builder):
        """После reload_configs кэши сбрасываются и данные перечитываются."""
        # Прогреваем кэши
        builder.get_domain_config("blog")
        builder.get_intent_config("storytelling")
        builder.get_overlay_config("infostyle")
        builder.get_output_format("text_only")

        # Сбрасываем
        builder.reload_configs()

        with patch('src.prompt_builder.load_domain_config', wraps=load_domain_config) as mock_load_domain, \
             patch('src.prompt_builder.load_intent_config', wraps=load_intent_config) as mock_load_intent, \
             patch('src.prompt_builder.load_overlay_config', wraps=load_overlay_config) as mock_load_overlay, \
             patch('src.prompt_builder.load_output_format', wraps=load_output_format) as mock_load_output:
            builder.get_domain_config("blog")
            builder.get_intent_config("storytelling")
            builder.get_overlay_config("infostyle")
            builder.get_output_format("text_only")
            assert mock_load_domain.call_count == 1
            assert mock_load_intent.call_count == 1
            assert mock_load_overlay.call_count == 1
            assert mock_load_output.call_count == 1

    def test_get_knowledge_base_caches(self, builder):
        """get_knowledge_base должен использовать внутренний _kb_cache."""
        # Мокаем load_knowledge_base, а не FileCache.get_or_load_multi
        with patch('src.prompt_builder.load_knowledge_base', wraps=load_knowledge_base) as mock_load_kb:
            builder.get_knowledge_base({"blog", "storytelling"}, "storytelling")
            builder.get_knowledge_base({"blog", "storytelling"}, "storytelling")
            # Ожидаем, что load_knowledge_base вызван только один раз (кэширование)
            assert mock_load_kb.call_count == 1

    def test_reload_configs_clears_kb_cache(self, builder):
        """reload_configs должен очищать _kb_cache."""
        builder.get_knowledge_base({"blog"}, None)
        # Подменяем метод clear
        with patch.object(builder._kb_cache, 'clear', wraps=builder._kb_cache.clear) as mock_clear:
            builder.reload_configs()
            mock_clear.assert_called_once()