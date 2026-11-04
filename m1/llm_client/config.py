"""Configuración de la aplicación, cargada desde variables de entorno o `.env`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Self, get_args

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from .anthropic_client import DEFAULT_ANTHROPIC_MODEL
from .exceptions import LLMConfigurationError
from .openai_client import DEFAULT_OPENAI_MODEL

ProviderName = Literal["openai", "anthropic"]
SUPPORTED_PROVIDERS: tuple[ProviderName, ...] = get_args(ProviderName)

# Variable de entorno -> campo de `Settings`.
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
    """Configuración validada.

    Las API keys se guardan como `SecretStr`: no aparecen en `repr`, `str` ni
    en los mensajes de error de Pydantic, así que no se filtran a los logs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderName = "openai"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = Field(default=DEFAULT_OPENAI_MODEL, min_length=1)
    anthropic_model: str = Field(default=DEFAULT_ANTHROPIC_MODEL, min_length=1)
    max_retries: int = Field(default=2, ge=0, le=10)
    timeout_s: float = Field(default=60.0, gt=0)

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Self:
        """Construye la configuración a partir del entorno.

        Si `env_file` existe, se carga antes; las variables ya definidas en el
        entorno tienen prioridad sobre las del archivo. Las variables vacías se
        tratan como no definidas.
        """
        if env_file is not None:
            load_dotenv(env_file)
        values = {
            field: raw.strip()
            for env_var, field in _ENV_VARS.items()
            if (raw := os.getenv(env_var)) and raw.strip()
        }
        return cls.model_validate(values)

    def api_key_for(self, provider: ProviderName) -> SecretStr:
        """Devuelve la API key del proveedor o lanza un error de configuración claro."""
        key = self.openai_api_key if provider == "openai" else self.anthropic_api_key
        if key is None or not key.get_secret_value():
            env_var = f"{provider.upper()}_API_KEY"
            raise LLMConfigurationError(
                f"Falta la API key. Definí {env_var} en el entorno o en .env.",
                provider=provider,
            )
        return key

    def model_for(self, provider: ProviderName) -> str:
        return self.openai_model if provider == "openai" else self.anthropic_model

    def configured_providers(self) -> list[ProviderName]:
        """Proveedores que tienen API key configurada."""
        return [
            p
            for p, key in (("openai", self.openai_api_key), ("anthropic", self.anthropic_api_key))
            if key is not None and key.get_secret_value()
        ]
