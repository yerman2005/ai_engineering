"""Pipeline de extracción de entidades técnicas con LangChain (LCEL).

Cadena: `prompt | model.with_structured_output(ExtraccionTecnica) | validación`,
envuelta en `.with_retry()` para reintentar si el LLM devuelve un JSON mal
formado o incompleto.
"""

from __future__ import annotations

import logging
import os
from functools import cache
from pathlib import Path
from typing import Any, Literal, Self

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from schemas import ExtraccionTecnica

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 1024
MAX_INTENTOS = 3

# Motivos de corte por falta de tokens según el proveedor (`finish_reason` / `stop_reason`).
_MOTIVOS_DE_CORTE = {"length", "max_tokens"}


# --------------------------------------------------------------------------- #
# Configuración del cliente (misma lógica que el Módulo 1)
# --------------------------------------------------------------------------- #

_ENV_VARS: dict[str, str] = {
    "LLM_PROVIDER": "provider",
    "OPENAI_API_KEY": "openai_api_key",
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "OPENAI_MODEL": "openai_model",
    "ANTHROPIC_MODEL": "anthropic_model",
    "LLM_MAX_RETRIES": "max_retries",
    "LLM_TIMEOUT_S": "timeout_s",
}


class Settings(BaseModel):
    """Configuración validada. Las API keys se guardan como `SecretStr` para no filtrarlas."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = Field(default=DEFAULT_OPENAI_MODEL, min_length=1)
    anthropic_model: str = Field(default=DEFAULT_ANTHROPIC_MODEL, min_length=1)
    max_retries: int = Field(default=2, ge=0, le=10)
    timeout_s: float = Field(default=60.0, gt=0)

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Self:
        """Construye la configuración desde el entorno (y `.env`, si existe)."""
        if env_file is not None:
            load_dotenv(env_file)
        values = {
            field: raw.strip().lower() if field == "provider" else raw.strip()
            for env_var, field in _ENV_VARS.items()
            if (raw := os.getenv(env_var)) and raw.strip()
        }
        return cls.model_validate(values)

    def api_key(self) -> SecretStr:
        """Devuelve la API key del proveedor configurado o lanza un error claro."""
        key = self.openai_api_key if self.provider == "openai" else self.anthropic_api_key
        if key is None or not key.get_secret_value():
            raise ValueError(f"Falta la API key. Definí {self.provider.upper()}_API_KEY en .env.")
        return key


def build_model(settings: Settings) -> BaseChatModel:
    """Crea el cliente ChatOpenAI o ChatAnthropic según `LLM_PROVIDER`."""
    if settings.provider == "openai":
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.api_key(),
            temperature=0,
            max_tokens=MAX_TOKENS,
            max_retries=settings.max_retries,
            timeout=settings.timeout_s,
        )
    # Sin `temperature`: la API actual de Anthropic rechaza los parámetros de muestreo.
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.api_key(),
        max_tokens=MAX_TOKENS,
        max_retries=settings.max_retries,
        timeout=settings.timeout_s,
    )


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

INSTRUCCIONES_FORMATO = (
    "- 'tecnologias': lista no vacía con el nombre propio de cada tecnología mencionada.\n"
    "- 'nivel_de_criticidad': exactamente uno de 'baja', 'media' o 'alta'.\n"
    "- 'resumen_tecnico': una o dos oraciones en español, sin inventar datos."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un ingeniero de software senior. Extraé la información técnica del texto "
            "que te pase el usuario (un log de error o una descripción de arquitectura).\n"
            "Respetá estas instrucciones de formato:\n{instrucciones_formato}",
        ),
        ("human", "{texto}"),
    ]
).partial(instrucciones_formato=INSTRUCCIONES_FORMATO)


# --------------------------------------------------------------------------- #
# Cadena LCEL
# --------------------------------------------------------------------------- #


class SalidaIncompletaError(Exception):
    """El LLM devolvió un JSON mal formado, inválido o cortado por falta de tokens."""


def _motivo_de_corte(raw: Any) -> str | None:
    """Lee `finish_reason` (OpenAI) o `stop_reason` (Anthropic) de la respuesta cruda."""
    metadata = getattr(raw, "response_metadata", None) or {}
    return metadata.get("finish_reason") or metadata.get("stop_reason")


def _validar_salida(resultado: dict[str, Any]) -> ExtraccionTecnica:
    """Verifica la salida estructurada antes de entregarla.

    Lanza `SalidaIncompletaError` (que dispara un reintento) si la respuesta se
    cortó por tokens o si no pudo parsearse/validarse contra el esquema.
    """
    motivo = _motivo_de_corte(resultado.get("raw"))
    if motivo in _MOTIVOS_DE_CORTE:
        logger.warning("Respuesta cortada por falta de tokens (motivo=%s). Se reintenta.", motivo)
        raise SalidaIncompletaError(f"respuesta incompleta (motivo de corte: {motivo})")

    error = resultado.get("parsing_error")
    parsed = resultado.get("parsed")
    if error is not None or parsed is None:
        logger.warning("Validación fallida: %s. Se reintenta.", error or "sin salida estructurada")
        raise SalidaIncompletaError(f"JSON mal formado o incompleto: {error}")

    if not isinstance(parsed, ExtraccionTecnica):
        parsed = ExtraccionTecnica.model_validate(parsed)
    logger.info("Validación OK (motivo de fin=%s).", motivo)
    return parsed


def build_chain(model: BaseChatModel) -> Runnable[dict[str, str], ExtraccionTecnica]:
    """Ensambla la cadena LCEL con salida estructurada y reintentos."""
    chain = (
        PROMPT
        # `json_schema`: salida estructurada nativa del proveedor (OpenAI y Anthropic).
        | model.with_structured_output(ExtraccionTecnica, method="json_schema", include_raw=True)
        | RunnableLambda(_validar_salida)
    )
    return chain.with_retry(
        retry_if_exception_type=(SalidaIncompletaError, ValidationError, OutputParserException),
        wait_exponential_jitter=True,
        stop_after_attempt=MAX_INTENTOS,
    )


@cache
def get_chain() -> Runnable[dict[str, str], ExtraccionTecnica]:
    """Construye (una sola vez) la cadena a partir de la configuración del entorno."""
    settings = Settings.from_env()
    logger.info("Proveedor configurado: %s", settings.provider)
    return build_chain(build_model(settings))


async def process_text(text: str) -> ExtraccionTecnica:
    """Ejecuta el pipeline sobre `text` y devuelve el objeto validado."""
    if not text or not text.strip():
        raise ValueError("El texto de entrada no puede estar vacío.")

    logger.info("Procesando texto (%d caracteres)...", len(text))
    try:
        resultado = await get_chain().ainvoke({"texto": text})
    except (SalidaIncompletaError, ValidationError, OutputParserException) as exc:
        logger.error("Sin salida válida tras %d intentos: %s", MAX_INTENTOS, exc)
        raise
    logger.info("Extracción completada: %s", resultado.model_dump(mode="json"))
    return resultado
