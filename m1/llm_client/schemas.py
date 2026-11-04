"""Esquemas Pydantic compartidos por todos los proveedores.

Definir estos modelos primero evita pasar diccionarios anidados entre capas:
cada dato que entra o sale del cliente está tipado y validado.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    """Roles admitidos en una conversación."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación, independiente del proveedor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str = Field(min_length=1, description="Texto del mensaje; no puede estar vacío.")


class GenerationConfig(BaseModel):
    """Parámetros de generación validados antes de llegar al SDK."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str | None = Field(
        default=None,
        min_length=1,
        description="Modelo a usar. Si es None se usa el modelo por defecto del proveedor.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Aleatoriedad de la respuesta (0 a 2). None usa el valor del proveedor.",
    )
    max_tokens: int = Field(default=1024, gt=0, le=128_000)
    timeout_s: float = Field(
        default=60.0,
        gt=0,
        description="Tiempo máximo total de la llamada, en segundos.",
    )


class TokenUsage(BaseModel):
    """Consumo de tokens informado por el proveedor."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(BaseModel):
    """Respuesta normalizada: la misma forma para OpenAI y Anthropic."""

    model_config = ConfigDict(frozen=True)

    content: str
    provider: str
    model: str
    usage: TokenUsage | None = None
    finish_reason: str | None = None


class ErrorInfo(BaseModel):
    """Error controlado que se devuelve en lugar de propagar la excepción."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    provider: str
    retryable: bool = False


class LLMResult(BaseModel):
    """Resultado de una generación: contiene una respuesta o un error, nunca ambos."""

    model_config = ConfigDict(frozen=True)

    response: ModelResponse | None = None
    error: ErrorInfo | None = None

    @property
    def ok(self) -> bool:
        return self.response is not None
