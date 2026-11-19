"""Configuración compartida: LLM (misma lógica que el Módulo 2), embeddings y ChromaDB."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Self

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, ConfigDict, Field, SecretStr

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ENV_FILE = BASE_DIR / ".env"

# Base vectorial: instalación local de ChromaDB en <repo>/chromadb-local.
CHROMA_DIR = BASE_DIR.parent / "chromadb-local" / "vectorstore"
COLLECTION_PREFIX = "m3_rag"

# Embeddings: se usa el mismo modelo para indexar y para consultar.
# - local: all-MiniLM-L6-v2 (ONNX, incluido en chromadb). Corre en la CPU, sin API key ni costo.
# - openai: text-embedding-3-small (requiere OPENAI_API_KEY con crédito).
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking (en tokens) y recuperación.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 1024


_ENV_VARS: dict[str, str] = {
    "LLM_PROVIDER": "provider",
    "EMBEDDINGS_PROVIDER": "embeddings_provider",
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
    embeddings_provider: Literal["local", "openai"] = "local"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = Field(default=DEFAULT_OPENAI_MODEL, min_length=1)
    anthropic_model: str = Field(default=DEFAULT_ANTHROPIC_MODEL, min_length=1)
    max_retries: int = Field(default=2, ge=0, le=10)
    timeout_s: float = Field(default=60.0, gt=0)

    @classmethod
    def from_env(cls, env_file: str | Path | None = ENV_FILE) -> Self:
        """Construye la configuración desde el entorno (y `.env`, si existe)."""
        if env_file is not None:
            load_dotenv(env_file)
        values = {
            field: raw.strip().lower() if field.endswith("provider") else raw.strip()
            for env_var, field in _ENV_VARS.items()
            if (raw := os.getenv(env_var)) and raw.strip()
        }
        return cls.model_validate(values)

    @property
    def collection_name(self) -> str:
        """Una colección por modelo de embeddings: los vectores de modelos distintos no se mezclan."""
        return f"{COLLECTION_PREFIX}_{self.embeddings_provider}"

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


class LocalEmbeddings(Embeddings):
    """Adaptador LangChain del modelo all-MiniLM-L6-v2 que trae ChromaDB (ONNX, 384 dimensiones).

    La primera vez descarga el modelo (~80 MB) a la caché de ChromaDB; después funciona offline.
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._modelo = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._modelo(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(settings: Settings) -> Embeddings:
    """Modelo de embeddings según `EMBEDDINGS_PROVIDER` (por defecto, local)."""
    if settings.embeddings_provider == "local":
        return LocalEmbeddings()

    key = settings.openai_api_key
    if key is None or not key.get_secret_value():
        raise ValueError("Falta OPENAI_API_KEY en .env (EMBEDDINGS_PROVIDER=openai).")
    return OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=key,
        max_retries=settings.max_retries,
        timeout=settings.timeout_s,
    )
