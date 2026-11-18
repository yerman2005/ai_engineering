"""Script de prueba: ingesta (si hace falta) y dos consultas al sistema RAG.

Uso:
    python main.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

from ingest import ingestar
from rag import get_rag_response

PREGUNTA = "¿Para qué sirve un asyncio.Semaphore cuando se llama a un LLM?"

# Pregunta trampa: la respuesta no está en los documentos.
PREGUNTA_TRAMPA = "¿Cuál es la capital de Australia?"


async def ejecutar(titulo: str, pregunta: str) -> bool:
    print(f"\n{'=' * 60}\n{titulo}\n{'=' * 60}\nPregunta: {pregunta}\n")
    try:
        resultado = await get_rag_response(pregunta)
    except Exception as exc:  # se muestra el error en lugar de cortar el script
        print(f"[error] {type(exc).__name__}: {exc}")
        return False
    print(resultado.model_dump_json(indent=2))
    return True


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Silencia los logs de HTTP para que se vean los del sistema RAG.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    await asyncio.to_thread(ingestar)
    ok = await ejecutar("Prueba 1: pregunta respondible con los documentos", PREGUNTA)
    ok_trampa = await ejecutar("Prueba 2: pregunta trampa (fuera de contexto)", PREGUNTA_TRAMPA)
    return 0 if ok and ok_trampa else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
