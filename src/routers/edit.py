"""
routers.edit

POST /api/edit — основной эндпоинт редактирования текста.

Выделено из src.main в итерации 8 дорожной карты.

NOTE: no `from __future__ import annotations` here. That future
feature turns annotations into strings; FastAPI then tries to
resolve them via get_type_hints on the endpoint object, but the
outer @limiter.limit(...) decorator returns a wrapper whose
__globals__ do not contain the request/response types (EditRequest,
EditResponse). As a result, endpoint registration fails with
NameError: 'EditRequest' is not defined. Do not re-enable the
future import in this module.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError

from src.auth import verify_api_key
from src.contracts import EditRequest, EditResponse
from src.error_mapping import (
    InvalidLLMOutputError,
    _llm_error_to_http_exception,
)
from src.llm_client import LLMError, LLMFallbackError
from src.prompt_builder import PromptBuilder
from src.rate_limit import RATE_LIMIT, limiter
from src.services.edit_service import (
    _build_audience_from_request,
    _build_dry_run_response,
    _generate_clean_edit,
    _log_edit_request_meta,
)
from src.shared_contracts import ALLOWED_PROVIDERS

logger = logging.getLogger(__name__)

router = APIRouter()


def get_prompt_builder(request: Request) -> PromptBuilder:
    prompt_builder = getattr(request.app.state, "prompt_builder", None)
    if prompt_builder is None:
        raise RuntimeError("PromptBuilder is not initialized")
    return prompt_builder


@router.post(
    "/api/edit",
    response_model=EditResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT)
async def edit_text(request: Request, body: EditRequest) -> EditResponse:
    try:
        audience = _build_audience_from_request(body)
        prompt_builder = get_prompt_builder(request)

        prompt, retrieval_meta = prompt_builder.build(
            text=body.text,
            domain=body.domain,
            intent=body.intent,
            audience=audience,
            overlays=body.overlays,
            output_mode=body.output_mode,
            include_knowledge=body.include_knowledge,
            include_few_shot=body.include_few_shot,
            include_retrieval_meta=True,
            deep_semantic_search=body.deep_semantic_search,
        )

        if body.dry_run:
            return _build_dry_run_response(body, prompt, retrieval_meta)

        providers_to_try = [body.provider] + [
            provider for provider in sorted(ALLOWED_PROVIDERS) if provider != body.provider
        ]

        response, edited_text, report = await _generate_clean_edit(
            prompt=prompt,
            providers=providers_to_try,
            body=body,
        )

        _log_edit_request_meta(body, retrieval_meta)
        return EditResponse(
            edited_text=edited_text,
            report=report,
            model=response.model,
            provider=response.provider,
            dry_run=False,
            usage={"tokens_used": response.tokens_used},
            raw_response={
                "finish_reason": response.finish_reason,
            },
            retrieval_meta=(retrieval_meta if body.include_retrieval_meta else None),
        )

    except InvalidLLMOutputError as error:
        logger.error("Output guard blocked response: %s", error.reasons)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=("The editor could not produce a valid formatted result. " "Please try again."),
        ) from error
    except LLMError as error:
        if isinstance(error, LLMFallbackError):
            logger.warning(
                "LLMFallbackError: provider=%s kind=%s upstream_status=%s "
                "skipped=%s unknown=%s prompt_length=%d",
                error.provider,
                error.kind,
                error.upstream_status,
                error.skipped_providers,
                error.unknown_providers,
                error.prompt_length,
            )
        else:
            logger.error("LLM error: %s", error, exc_info=True)
        raise _llm_error_to_http_exception(error) from error
    except FileNotFoundError as error:
        logger.error("Config file not found: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service configuration error. Contact support.",
        ) from error
    except HTTPException:
        raise
    except ValidationError as error:
        logger.error("Validation error: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error.errors(),
        ) from error
    except Exception as error:
        logger.error("Unexpected error: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from error
