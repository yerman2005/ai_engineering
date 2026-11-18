"""Contrato de datos del sistema RAG: respuesta validada con Pydantic."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RespuestaRAG(BaseModel):
    """Respuesta fundamentada en el contexto recuperado, con sus fuentes."""

    model_config = ConfigDict(extra="forbid")

    respuesta: str = Field(
        min_length=1,
        description=(
            "Respuesta en español basada exclusivamente en el CONTEXTO. "
            "Si la información no está en el contexto, exactamente: 'No lo sé'."
        ),
    )
    referencias: list[str] = Field(
        default_factory=list,
        description=(
            "Nombres de los archivos fuente (los indicados como [fuente: ...]) usados para "
            "responder. Lista vacía si la respuesta es 'No lo sé'."
        ),
    )

    @field_validator("respuesta")
    @classmethod
    def _respuesta_no_vacia(cls, valor: str) -> str:
        valor = valor.strip()
        if not valor:
            raise ValueError("la respuesta no puede estar vacía")
        return valor
