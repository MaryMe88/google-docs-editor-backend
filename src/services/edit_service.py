"""
services.edit_service

Helpers для /api/edit: логирование метаданных, разбор вывода LLM,
валидация, сборка dry-run ответа и orchestration вызова LLM.

Выделено из src.main в итерации 8 дорожной карты.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.config_types import AudienceProfile
from src.contracts import EditRequest, EditResponse
from src.error_mapping import InvalidLLMOutputError
from src.llm_client import call_with_fallback
from src.output_guard import (
    find_placeholder_leaks,
    harden_prompt_against_placeholders,
    has_placeholder_leak,
)

logger = logging.getLogger(__name__)


_MARKER_TEXT = re.compile(r"={2,}\s*ТЕКСТ\s*={2,}", re.IGNORECASE)
_MARKER_REPORT = re.compile(r"={2,}\s*ОТЧЁТ\s*={2,}", re.IGNORECASE)


def _log_edit_request_meta(body: EditRequest, retrieval_meta: dict | None = None) -> None:
    log_data = {
        "event": "edit_request",
        "domain": body.domain,
        "intent": body.intent,
        "overlays": body.overlays,
        "provider": body.provider,
        "output_mode": body.output_mode,
        "dry_run": body.dry_run,
        "text_length": len(body.text),
        "include_knowledge": body.include_knowledge,
        "include_few_shot": body.include_few_shot,
    }
    if retrieval_meta:
        log_data["retrieval_meta"] = retrieval_meta
    logger.info(json.dumps(log_data, ensure_ascii=False))


def _split_edit_output(raw: str, output_mode: str) -> tuple[str, str | None]:
    if output_mode == "text_and_report":
        return _parse_text_and_report(raw)
    return raw, None


def _looks_like_report_instead_of_text(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True

    report_signals = [
        'count the "не x, а y" occurrences',
        "count the",
        "so we have",
        "this is a marker",
        "also, the use of",
        "маркеры:",
        "исходный ип:",
        "итоговый ип:",
        "нужен второй проход",
    ]
    return any(signal in normalized for signal in report_signals)


def _validate_edit_output(
    *,
    raw_content: str,
    edited_text: str,
    report: str | None,
    output_mode: str,
) -> list[str]:
    reasons: list[str] = []

    if has_placeholder_leak(edited_text):
        reasons.extend(find_placeholder_leaks(edited_text))

    if output_mode == "text_and_report":
        has_text_marker = _MARKER_TEXT.search(raw_content) is not None

        if has_text_marker and not edited_text.strip():
            reasons.append("EMPTY_TEXT_BLOCK")

        if not has_text_marker and _looks_like_report_instead_of_text(edited_text):
            reasons.append("REPORT_INSTEAD_OF_TEXT")

        if has_placeholder_leak(report or ""):
            reasons.extend(find_placeholder_leaks(report or ""))

    return sorted(set(reasons))


async def _generate_clean_edit(
    prompt: str,
    providers: list[str],
    body: EditRequest,
) -> tuple[Any, str, str | None]:
    response = await call_with_fallback(
        prompt=prompt,
        providers=providers,
        model=body.model,
        temperature=body.temperature,
        max_retries_per_provider=2,
        source_text=body.text,
    )
    edited_text, report = _split_edit_output(response.content, body.output_mode)

    reasons = _validate_edit_output(
        raw_content=response.content,
        edited_text=edited_text,
        report=report,
        output_mode=body.output_mode,
    )
    if not reasons:
        return response, edited_text, report

    logger.warning(
        "Guard: ответ LLM невалиден, выполняем повторную попытку. Причины: %s",
        reasons,
    )

    hardened_prompt = harden_prompt_against_placeholders(prompt)
    hardened_prompt += (
        "\n\nКритично: строго соблюдай формат ответа. "
        "Если запрошен режим text_and_report, сначала выведи блок "
        "===ТЕКСТ=== с полным отредактированным текстом, затем блок "
        "===ОТЧЁТ===. Не выводи один только анализ, список маркеров или "
        "служебные пояснения вместо текста."
    )

    response = await call_with_fallback(
        prompt=hardened_prompt,
        providers=providers,
        model=body.model,
        temperature=min(body.temperature, 0.2),
        max_retries_per_provider=2,
        source_text=body.text,
    )
    edited_text, report = _split_edit_output(response.content, body.output_mode)

    reasons = _validate_edit_output(
        raw_content=response.content,
        edited_text=edited_text,
        report=report,
        output_mode=body.output_mode,
    )
    if reasons:
        logger.error(
            "Guard: ответ LLM остался невалидным после повторной попытки: %s",
            reasons,
        )
        raise InvalidLLMOutputError(reasons)

    return response, edited_text, report


def _build_audience_from_request(body: EditRequest) -> AudienceProfile | None:
    """Строит AudienceProfile из тела запроса, если он передан."""
    if body.audience is None:
        return None
    return AudienceProfile(
        kind=body.audience.kind,
        expertise=body.audience.expertise,
        formality=body.audience.formality,
        description=body.audience.description,
    )


def _build_dry_run_response(
    body: EditRequest,
    prompt: str,
    retrieval_meta: dict[str, Any],
) -> EditResponse:
    """Формирует ответ для dry_run."""
    _log_edit_request_meta(body, retrieval_meta)
    return EditResponse(
        edited_text=body.text,
        report=None,
        provider=body.provider,
        model=body.model,
        dry_run=True,
        usage={},
        raw_response={},
        retrieval_meta=retrieval_meta,
    )


def _parse_text_and_report(raw: str) -> tuple[str, str | None]:
    text_match = _MARKER_TEXT.search(raw)
    report_match = _MARKER_REPORT.search(raw)

    if text_match is None:
        logger.warning(
            "Маркер ТЕКСТ не найден в ответе LLM. "
            "Возвращаем весь ответ как текст. Длина: %d символов",
            len(raw),
        )
        return raw.strip(), None

    if report_match is None:
        edited_text = raw[text_match.end() :].strip()
        if not edited_text:
            logger.warning("Блок ТЕКСТ найден, но содержимое пусто.")
        return edited_text, None

    if text_match.start() < report_match.start():
        edited_text = raw[text_match.end() : report_match.start()].strip()
        report = raw[report_match.end() :].strip()
    else:
        logger.warning(
            "Маркеры ТЕКСТ/ОТЧЁТ идут в перевёрнутом порядке — "
            "разбираем с учётом этого, отчёт сохраняется."
        )
        edited_text = raw[text_match.end() :].strip()
        report = raw[report_match.end() : text_match.start()].strip()

    if not edited_text:
        logger.warning("Блок ТЕКСТ найден, но содержимое пусто.")

    return edited_text, (report or None)
