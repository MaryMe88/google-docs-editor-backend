from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class LLMError(RuntimeError):
    pass


class LLMProvider(str, Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str
    model: str
    usage: Dict[str, Any]
    raw_response: Dict[str, Any]


class OpenAICompatibleClient:
    def __init__(self, provider: LLMProvider, api_key: Optional[str] = None) -> None:
        self.provider = provider
        self.api_key = api_key or os.getenv(self._env_var_name(provider), "")
        self.base_url = self._base_url(provider)
        if not self.api_key:
            raise LLMError(
                f"API key for provider '{provider.value}' is missing. "
                f"Set {self._env_var_name(provider)} or pass api_key explicitly."
            )

    @staticmethod
    def _env_var_name(provider: LLMProvider) -> str:
        mapping = {
            LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.PERPLEXITY: "PERPLEXITY_API_KEY",
        }
        return mapping[provider]

    @staticmethod
    def _base_url(provider: LLMProvider) -> str:
        mapping = {
            LLMProvider.OPENROUTER: "https://openrouter.ai/api/v1/chat/completions",
            LLMProvider.OPENAI: "https://api.openai.com/v1/chat/completions",
            LLMProvider.PERPLEXITY: "https://api.perplexity.ai/chat/completions",
        }
        return mapping[provider]

    @staticmethod
    def _default_model(provider: LLMProvider) -> str:
        mapping = {
            LLMProvider.OPENROUTER: "openrouter/auto",
            LLMProvider.OPENAI: "gpt-4.1-mini",
            LLMProvider.PERPLEXITY: "sonar",
        }
        return mapping[provider]

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        timeout: int = 90,
    ) -> LLMResult:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self._default_model(self.provider),
            "messages": messages,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider is LLMProvider.OPENROUTER:
            headers["HTTP-Referer"] = "https://localhost"
            headers["X-Title"] = "text-editor-api"

        request = urllib.request.Request(
            self.base_url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"HTTP {exc.code} from {self.provider.value}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Network error while calling {self.provider.value}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"Invalid JSON from {self.provider.value}: {exc}") from exc

        text = self._extract_text(parsed)
        model_name = str(parsed.get("model") or payload["model"])
        usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
        return LLMResult(
            text=text,
            provider=self.provider.value,
            model=model_name,
            usage=usage,
            raw_response=parsed,
        )

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError("LLM response does not contain choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMError("LLM response choice is malformed")
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                chunks: List[str] = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        chunks.append(item["text"])
                if chunks:
                    return "\n".join(chunks)
        text = first.get("text")
        if isinstance(text, str):
            return text
        raise LLMError("LLM response does not contain text content")



def create_llm_client(provider: str, api_key: Optional[str] = None) -> OpenAICompatibleClient:
    normalized = provider.strip().lower()
    try:
        enum_provider = LLMProvider(normalized)
    except ValueError as exc:
        raise LLMError(f"Unsupported provider: {provider}") from exc
    return OpenAICompatibleClient(enum_provider, api_key=api_key)
