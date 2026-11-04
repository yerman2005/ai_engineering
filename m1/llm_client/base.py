"""Contrato común de todos los clientes LLM (abstracción)."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from types import TracebackType
from typing import ClassVar, Self

from .exceptions import LLMConfigurationError, LLMError, LLMTimeoutError
from .schemas import ChatMessage, GenerationConfig, ModelResponse


class BaseLLMClient(ABC):
    """Clase base de los clientes asíncronos.

    Implementa el patrón *template method*: los métodos públicos `generate` y
    `stream` aplican el timeout y traducen los errores. Cada subclase solo
    implementa la llamada concreta a su SDK (`_generate` / `_stream`) y la
    traducción de excepciones (`_map_error`).
    """

    #: Nombre del proveedor, usado en respuestas y errores.
    provider_name: ClassVar[str]
    #: Excepciones del SDK que `_map_error` sabe traducir.
    sdk_error_types: ClassVar[tuple[type[Exception], ...]]

    def __init__(self, *, default_model: str) -> None:
        self.default_model = default_model

    # ------------------------------------------------------------------ API pública

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> ModelResponse:
        """Genera una respuesta completa.

        Raises:
            LLMError: ante cualquier fallo del proveedor, ya normalizado.
        """
        config = config or GenerationConfig()
        self._check_messages(messages)
        model = config.model or self.default_model
        try:
            async with asyncio.timeout(config.timeout_s):
                return await self._generate(messages, model, config)
        except TimeoutError as exc:
            raise LLMTimeoutError(
                f"Sin respuesta después de {config.timeout_s}s.",
                provider=self.provider_name,
            ) from exc
        except self.sdk_error_types as exc:
            raise self._map_error(exc) from exc

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Devuelve un generador asíncrono con los fragmentos de texto a medida que llegan.

        En streaming, `config.timeout_s` es el tiempo máximo de espera entre un
        fragmento y el siguiente (incluido el primero), no el total de la respuesta.

        Raises:
            LLMError: ante cualquier fallo del proveedor, ya normalizado.
        """
        config = config or GenerationConfig()
        self._check_messages(messages)
        model = config.model or self.default_model
        try:
            # `aclosing` garantiza que el stream del SDK se cierre aunque el
            # consumidor deje de iterar antes de terminar.
            async with aclosing(self._stream(messages, model, config)) as chunks:
                while True:
                    try:
                        async with asyncio.timeout(config.timeout_s):
                            chunk = await anext(chunks)
                    except StopAsyncIteration:
                        return
                    yield chunk
        except TimeoutError as exc:
            raise LLMTimeoutError(
                f"El stream no envió datos durante {config.timeout_s}s.",
                provider=self.provider_name,
            ) from exc
        except self.sdk_error_types as exc:
            raise self._map_error(exc) from exc

    @abstractmethod
    async def aclose(self) -> None:
        """Libera las conexiones HTTP del SDK."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------- A implementar por subclases

    @abstractmethod
    async def _generate(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        config: GenerationConfig,
    ) -> ModelResponse:
        """Llamada no streaming al SDK del proveedor."""

    @abstractmethod
    def _stream(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        """Generador asíncrono que recorre el stream del SDK y produce texto."""

    @abstractmethod
    def _map_error(self, exc: Exception) -> LLMError:
        """Traduce una excepción del SDK a la jerarquía `LLMError`."""

    # ----------------------------------------------------------------------- Helpers

    def _check_messages(self, messages: Sequence[ChatMessage]) -> None:
        if not messages:
            raise LLMConfigurationError(
                "Se necesita al menos un mensaje.", provider=self.provider_name
            )
