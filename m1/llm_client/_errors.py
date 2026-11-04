"""Traducción de excepciones de los SDKs oficiales a la jerarquía del dominio.

`openai` y `anthropic` exponen las mismas clases de error (ambos SDKs se generan
con la misma herramienta), así que una sola función sirve para los dos.
"""

from __future__ import annotations

from types import ModuleType

from .exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)


def map_sdk_error(exc: Exception, *, sdk: ModuleType, provider: str) -> LLMError:
    """Convierte una excepción del SDK `sdk` en un `LLMError`.

    El orden importa: `APITimeoutError` hereda de `APIConnectionError`, y las
    subclases de estado HTTP heredan de `APIStatusError`.
    """
    message = getattr(exc, "message", None) or str(exc)

    if isinstance(exc, sdk.APITimeoutError):
        return LLMTimeoutError(message, provider=provider)
    if isinstance(exc, sdk.APIConnectionError):
        return LLMConnectionError(message, provider=provider)
    if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return LLMAuthenticationError(message, provider=provider)
    if isinstance(exc, sdk.RateLimitError):
        return LLMRateLimitError(message, provider=provider)
    if isinstance(exc, sdk.APIStatusError):
        return LLMProviderError(message, provider=provider, status_code=exc.status_code)
    return LLMProviderError(message, provider=provider)
