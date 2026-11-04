"""Excepciones del dominio.

Los clientes traducen las excepciones propias de cada SDK a esta jerarquía,
de modo que el resto de la aplicación nunca dependa de `openai` o `anthropic`.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base de todos los errores del cliente LLM."""

    code: str = "llm_error"
    retryable: bool = False

    def __init__(self, message: str, *, provider: str) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider

    def __str__(self) -> str:
        return f"[{self.provider}] {self.code}: {self.message}"


class LLMConfigurationError(LLMError):
    """Configuración inválida (proveedor desconocido, falta la API key, etc.)."""

    code = "configuration_error"


class LLMAuthenticationError(LLMError):
    """API key inválida o sin permisos."""

    code = "authentication_error"


class LLMRateLimitError(LLMError):
    """Se superó el límite de tasa o la cuota del proveedor (HTTP 429)."""

    code = "rate_limit_error"
    retryable = True


class LLMConnectionError(LLMError):
    """Fallo de red antes de recibir respuesta."""

    code = "connection_error"
    retryable = True


class LLMTimeoutError(LLMError):
    """La llamada superó el tiempo máximo permitido."""

    code = "timeout_error"
    retryable = True


class LLMProviderError(LLMError):
    """Cualquier otro error devuelto por la API del proveedor."""

    code = "provider_error"

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.status_code = status_code
        # Los errores 5xx son transitorios y pueden reintentarse.
        self.retryable = status_code is not None and status_code >= 500
