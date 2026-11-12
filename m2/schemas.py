"""Contrato de datos del pipeline: estructura de salida validada con Pydantic."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NivelDeCriticidad(StrEnum):
    """Nivel de criticidad técnica del texto analizado."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto (log de error o descripción de arquitectura).

    Las descripciones de cada campo forman parte del esquema que recibe el modelo
    a través de `with_structured_output()`.
    """

    model_config = ConfigDict(extra="forbid")

    tecnologias: list[str] = Field(
        min_length=1,
        description=(
            "Tecnologías, lenguajes, frameworks, bases de datos o servicios mencionados "
            "en el texto, con su nombre propio (ej.: 'FastAPI', 'Redis', 'PostgreSQL')."
        ),
    )
    nivel_de_criticidad: NivelDeCriticidad = Field(
        description=(
            "Criticidad técnica: 'alta' si hay caída, pérdida de datos o impacto en "
            "producción; 'media' si hay degradación o riesgo; 'baja' si es informativo."
        ),
    )
    resumen_tecnico: str = Field(
        min_length=1,
        description="Resumen técnico breve (una o dos oraciones) en español.",
    )

    @field_validator("tecnologias")
    @classmethod
    def _limpiar_tecnologias(cls, valores: list[str]) -> list[str]:
        """Quita espacios, descarta vacíos y duplicados (sin distinguir mayúsculas)."""
        limpias: list[str] = []
        vistas: set[str] = set()
        for valor in valores:
            nombre = valor.strip()
            if nombre and nombre.casefold() not in vistas:
                vistas.add(nombre.casefold())
                limpias.append(nombre)
        if not limpias:
            raise ValueError("la lista de tecnologías no puede estar vacía")
        return limpias

    @field_validator("resumen_tecnico")
    @classmethod
    def _resumen_no_vacio(cls, valor: str) -> str:
        valor = valor.strip()
        if not valor:
            raise ValueError("el resumen técnico no puede estar vacío")
        return valor
