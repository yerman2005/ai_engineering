"""Implementación del cliente para Anthropic (Messages API)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from pydantic import SecretStr

from ._errors import map_sdk_error
from .base import BaseLLMClient
from .exceptions import LLMConfigurationError, LLMError
from .schemas import ChatMessage, GenerationConfig, ModelResponse, Role, TokenUsage

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Cliente asíncrono sobre `AsyncAnthropic`.

    Diferencias con OpenAI que esta clase oculta al resto de la aplicación:

    * El prompt de sistema va en el parámetro `system`, no como un mensaje más.
    * `response.content` es una lista de bloques; los modelos actuales pueden
      devolver bloques `thinking` antes del texto, así que se toman solo los `text`.
    * La API actual de Claude no acepta parámetros de muestreo (`temperature`):
      si se configura, se ignora y se deja constancia en el log.
    """

    provider_name = "anthropic"
    sdk_error_types = (anthropic.APIError,)

    def __init__(
        self,
        *,
        api_key: SecretStr,
        default_model: str = DEFAULT_ANTHROPIC_MODEL,
        max_retries: int = 2,
        sdk_client: AsyncAnthropic | None = None,
    ) -> None:
        super().__init__(default_model=default_model)
        # El SDK reintenta solo, con backoff exponencial, los errores 408/409/429/5xx
        # y de conexión. `sdk_client` permite inyectar un doble en los tests.
        self._client = sdk_client or AsyncAnthropic(
            api_key=api_key.get_secret_value(),
            max_retries=max_retries,
        )

    async def _generate(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        config: GenerationConfig,
    ) -> ModelResponse:
        response = await self._client.messages.create(**self._build_params(messages, model, config))
        text = "".join(block.text for block in response.content if block.type == "text")
        return ModelResponse(
            content=text,
            provider=self.provider_name,
            model=response.model,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            finish_reason=response.stop_reason,
        )

    async def _stream(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        # `messages.stream` es el helper del SDK: `text_stream` produce solo el
        # texto y descarta los eventos de otros tipos de bloque.
        async with self._client.messages.stream(
            **self._build_params(messages, model, config)
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def _map_error(self, exc: Exception) -> LLMError:
        return map_sdk_error(exc, sdk=anthropic, provider=self.provider_name)

    async def aclose(self) -> None:
        await self._client.close()

    def _build_params(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        config: GenerationConfig,
    ) -> dict[str, Any]:
        system_parts = [m.content for m in messages if m.role is Role.SYSTEM]
        conversation = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role is not Role.SYSTEM
        ]
        if not conversation:
            raise LLMConfigurationError(
                "Anthropic necesita al menos un mensaje de usuario además del de sistema.",
                provider=self.provider_name,
            )
        if config.temperature is not None:
            logger.warning(
                "Anthropic no admite 'temperature' en la API actual; se ignora el valor %s.",
                config.temperature,
            )

        params: dict[str, Any] = {
            "model": model,
            "max_tokens": config.max_tokens,
            "messages": conversation,
        }
        if system_parts:
            params["system"] = "\n\n".join(system_parts)
        return params
