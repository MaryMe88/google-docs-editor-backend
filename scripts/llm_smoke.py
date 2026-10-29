"""
llm_smoke.py

Ручной smoke-тест LLM-клиента. Требует реальный API-ключ (OPENROUTER_API_KEY)
и делает платный вызов — запускать только вручную.

Перенесено из src/llm_client.py (итерация 8.1 карты №2): demo-код вынесен
из модуля, чтобы случайный `python -m src.llm_client` не приводил к платному
вызову API в production-контексте.

Использование:
    $env:PYTHONPATH="."
    python scripts/llm_smoke.py
"""

from __future__ import annotations

import asyncio
import logging

from src.llm_client import call_with_fallback


async def _test_client() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    test_prompt = "Напиши короткий дружелюбный абзац о пользе хорошей редактуры."
    test_source = "Короткий текст для теста."

    try:
        response = await call_with_fallback(
            prompt=test_prompt,
            providers=["openrouter"],
            source_text=test_source,
            temperature=0.3,
            max_retries_per_provider=1,
        )
        print()
        print(response.content)
        print(f"\nModel: {response.model}")
        print(f"Provider: {response.provider}")
        print(f"Tokens used: {response.tokens_used}")
        print(f"Finish reason: {response.finish_reason}")
    except Exception as error:
        print(f"LLM error: {error}")


if __name__ == "__main__":
    asyncio.run(_test_client())
