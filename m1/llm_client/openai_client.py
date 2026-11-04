"""Implementación del cliente para OpenAI (Chat Completions)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import openai
from openai import AsyncOpenAI
from pydantic import SecretStr

from ._errors import map_sdk_error
from .base import BaseLLMClient
from .exceptions import LLMError
from .schemas import ChatMessage, GenerationConfig, ModelResponse, TokenUsage

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAIClient(BaseLLMClient):
    """Cliente asíncrono sobre `AsyncOpenAI`."""

    provider_name = "openai"
    sdk_error_types = (openai.APIError,)

    def __init__(
        self,
        *,
        api_key: SecretStr,
        default_model: str = DEFAULT_OPENAI_MODEL,
        max_retries: int = 2,
        sdk_client: AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(default_model=default_model)
        # El SDK reintenta solo, con backoff exponencial, los errores 408/409/429/5xx
        # y de conexión. `sdk_client` permite inyectar un doble en los tests.
        self._client = sdk_client or AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            max_retries=max_retries,
        )

    async def _generate(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        config: GenerationConfig,
    ) -> ModelResponse:
        response = await self._client.chat.completions.create(
            **self._build_params(messages, model, config)
        )
        choice = response.choices[0]
        usage = (
            TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            if response.usage
            else None
        )
        return ModelResponse(
            content=choice.message.content or "",
            provider=self.provider_name,
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    async def _stream(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            **self._build_params(messages, model, config),
            stream=True,
        )
        async with stream:
            async for chunk in stream:
                # Algunos chunks (p. ej. el final) no traen choices ni texto.
                if chunk.choices and (text := chunk.choices[0].delta.content):
                    yield text

    def _map_error(self, exc: Exception) -> LLMError:
        return map_sdk_error(exc, sdk=openai, provider=self.provider_name)

    async def aclose(self) -> None:
        await self._client.close()

    @staticmethod
    def _build_params(
        messages: Sequence[ChatMessage],
        model: str,
        config: GenerationConfig,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "max_completion_tokens": config.max_tokens,
        }
        if config.temperature is not None:
            params["temperature"] = config.temperature
        return params
