"""`AsyncLLMManager`: punto de entrada único para la lógica de negocio."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from types import TracebackType
from typing import Self

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .config import ProviderName, Settings
from .exceptions import LLMConfigurationError, LLMError
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ErrorInfo, GenerationConfig, LLMResult, Role

logger = logging.getLogger(__name__)

# Registro de proveedores (patrón Factory): el nombre en la configuración
# decide qué clase concreta se instancia.
_PROVIDERS: dict[ProviderName, type[OpenAIClient | AnthropicClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}

Prompt = str | Sequence[ChatMessage]


class AsyncLLMManager:
    """Fachada asíncrona sobre un `BaseLLMClient`.

    El código que la usa no sabe qué proveedor hay detrás. Responsabilidades:

    * Crear el cliente correcto a partir de la configuración (`from_settings`).
    * Aceptar un prompt simple o una lista de `ChatMessage`.
    * Convertir los errores en un `LLMResult` con `ErrorInfo` (modo normal), de
      modo que una falla no interrumpa el flujo del programa.
    * Limitar la concurrencia en lotes con un semáforo (`generate_many`).
    """

    def __init__(
        self,
        client: BaseLLMClient,
        *,
        default_config: GenerationConfig | None = None,
    ) -> None:
        self._client = client
        self._default_config = default_config or GenerationConfig()

    @classmethod
    def from_settings(cls, settings: Settings, provider: ProviderName | None = None) -> Self:
        """Crea el manager para `provider` (o el de `settings.provider` si es None).

        Raises:
            LLMConfigurationError: si el proveedor no existe o le falta la API key.
        """
        name = provider or settings.provider
        client_cls = _PROVIDERS.get(name)
        if client_cls is None:
            raise LLMConfigurationError(
                f"Proveedor no soportado. Opciones: {', '.join(_PROVIDERS)}.",
                provider=str(name),
            )
        client = client_cls(
            api_key=settings.api_key_for(name),
            default_model=settings.model_for(name),
            max_retries=settings.max_retries,
        )
        return cls(client, default_config=GenerationConfig(timeout_s=settings.timeout_s))

    @property
    def provider(self) -> str:
        return self._client.provider_name

    async def generate(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        config: GenerationConfig | None = None,
    ) -> LLMResult:
        """Genera una respuesta completa. Nunca lanza `LLMError`: lo devuelve en el resultado."""
        messages = self._build_messages(prompt, system)
        try:
            response = await self._client.generate(messages, config or self._default_config)
        except LLMError as exc:
            logger.warning("Falló la generación con %s: %s", self.provider, exc)
            return LLMResult(
                error=ErrorInfo(
                    code=exc.code,
                    message=exc.message,
                    provider=exc.provider,
                    retryable=exc.retryable,
                )
            )
        return LLMResult(response=response)

    async def stream(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Produce los fragmentos de texto a medida que llegan.

        Un stream puede cortarse a mitad de camino, así que los errores no se
        pueden devolver como valor: se lanzan como `LLMError` (nunca como
        excepciones del SDK) para que quien consume decida qué hacer.
        """
        messages = self._build_messages(prompt, system)
        async with aclosing(
            self._client.stream(messages, config or self._default_config)
        ) as chunks:
            async for chunk in chunks:
                yield chunk

    async def generate_many(
        self,
        prompts: Sequence[Prompt],
        *,
        system: str | None = None,
        config: GenerationConfig | None = None,
        max_concurrency: int = 5,
    ) -> list[LLMResult]:
        """Procesa varios prompts en paralelo, con a lo sumo `max_concurrency` llamadas activas.

        El semáforo evita disparar todas las llamadas a la vez y recibir errores
        429. Los resultados respetan el orden de `prompts`.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency debe ser al menos 1.")
        semaphore = asyncio.Semaphore(max_concurrency)

        async def run(prompt: Prompt) -> LLMResult:
            async with semaphore:
                return await self.generate(prompt, system=system, config=config)

        return list(await asyncio.gather(*(run(p) for p in prompts)))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    @staticmethod
    def _build_messages(prompt: Prompt, system: str | None) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role=Role.SYSTEM, content=system))
        if isinstance(prompt, str):
            messages.append(ChatMessage(role=Role.USER, content=prompt))
        else:
            messages.extend(prompt)
        return messages
