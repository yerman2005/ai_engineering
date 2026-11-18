"""Módulo de ingesta: lee `data/`, fragmenta los documentos y los persiste en ChromaDB.

Uso:
    python ingest.py
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    Settings,
    build_embeddings,
)

logger = logging.getLogger(__name__)

EXTENSIONES = {".txt", ".md"}


def get_vectorstore(settings: Settings | None = None) -> Chroma:
    """Abre (o crea) la colección persistente en disco."""
    settings = settings or Settings.from_env()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(settings),
        persist_directory=str(CHROMA_DIR),
    )


def cargar_documentos() -> list[Document]:
    """Lee los archivos .txt y .md de la carpeta `data/`."""
    archivos = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in EXTENSIONES)
    if not archivos:
        raise FileNotFoundError(f"No hay archivos .txt o .md en {DATA_DIR}")
    return [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
        for p in archivos
    ]


def fragmentar(documentos: list[Document]) -> list[Document]:
    """Divide los documentos en chunks de CHUNK_SIZE tokens con CHUNK_OVERLAP de solapamiento."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documentos)


def ingestar() -> int:
    """Puebla la colección si está vacía. Devuelve la cantidad de chunks en la colección."""
    vectorstore = get_vectorstore()
    existentes = vectorstore._collection.count()
    if existentes > 0:
        logger.info(
            "La colección '%s' ya existe en %s (%d chunks). No se reindexa.",
            COLLECTION_NAME, CHROMA_DIR, existentes,
        )
        return existentes

    documentos = cargar_documentos()
    chunks = fragmentar(documentos)

    # IDs deterministas: <archivo>-<n>.
    contador: dict[str, int] = {}
    ids: list[str] = []
    for chunk in chunks:
        fuente = chunk.metadata["source"]
        contador[fuente] = contador.get(fuente, 0) + 1
        ids.append(f"{fuente}-{contador[fuente]}")

    vectorstore.add_documents(chunks, ids=ids)
    logger.info(
        "Indexados %d chunks de %d documentos en '%s' (%s).",
        len(chunks), len(documentos), COLLECTION_NAME, CHROMA_DIR,
    )
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ingestar()
