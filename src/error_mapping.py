"""
error_mapping

Преобразование доменных ошибок приложения в HTTP-ответы.

Выделено из src.main в итерации 8 дорожной карты.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from src.llm_client import LLMError, LLMFallbackError


class InvalidLLMOutputError(Exception):
    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__(f"Invalid LLM output: {reasons}")


def _llm_error_to_http_exception(error: LLMError) -> HTTPException:
    if isinstance(error, LLMFallbackError):
        kind = error.kind
        if kind == "rate_limit":
            return HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="LLM provider rate limit reached. Please try again later.",
            )
        if kind == "context_limit":
            return HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The text or editing instructions are too large. Please shorten them.",
            )
        if kind in ("timeout", "upstream_error"):
            return HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service is temporarily unavailable. Please try again later.",
            )
        if kind in ("authentication", "configuration"):
            return HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service configuration is temporarily unavailable.",
            )
        if kind == "invalid_response":
            return HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM service returned an empty or invalid response. "
                "Please try again later.",
            )
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM service returned an invalid response. Please try again later.",
        )
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="LLM service returned an invalid response. Please try again later.",
    )
