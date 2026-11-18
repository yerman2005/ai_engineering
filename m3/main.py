"""Script de prueba: ingesta (si hace falta) y dos consultas al sistema RAG.

Uso:
    python main.py                                   # ambas preguntas por defecto
    python main.py "¿Qué es un embedding?"           # pregunta propia, trampa por defecto
    python main.py --trampa "¿Quién ganó el Mundial 86?"
    python main.py "¿Qué es un embedding?" -t "¿Cuál es la capital de Francia?"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from ingest import ingestar
from rag import get_rag_response

PREGUNTA = "¿Para qué sirve un asyncio.Semaphore cuando se llama a un LLM?"

# Pregunta trampa: la respuesta no está en los documentos.
PREGUNTA_TRAMPA = "¿Cuál es la capital de Australia?"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba del sistema RAG con dos consultas.")
    parser.add_argument(
        "pregunta",
        nargs="?",
        default=PREGUNTA,
        help=f"Pregunta respondible con los documentos (por defecto: {PREGUNTA!r}).",
    )
    parser.add_argument(
        "-t",
        "--trampa",
        default=PREGUNTA_TRAMPA,
        help=f"Pregunta trampa fuera de contexto (por defecto: {PREGUNTA_TRAMPA!r}).",
    )
    return parser.parse_args(argv)


async def ejecutar(titulo: str, pregunta: str) -> bool:
    print(f"\n{'=' * 60}\n{titulo}\n{'=' * 60}\nPregunta: {pregunta}\n")
    try:
        resultado = await get_rag_response(pregunta)
    except Exception as exc:  # se muestra el error en lugar de cortar el script
        print(f"[error] {type(exc).__name__}: {exc}")
        return False
    print(resultado.model_dump_json(indent=2))
    return True


async def main(pregunta: str = PREGUNTA, pregunta_trampa: str = PREGUNTA_TRAMPA) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Silencia los logs de HTTP para que se vean los del sistema RAG.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    await asyncio.to_thread(ingestar)
    ok = await ejecutar("Prueba 1: pregunta respondible con los documentos", pregunta)
    ok_trampa = await ejecutar("Prueba 2: pregunta trampa (fuera de contexto)", pregunta_trampa)
    return 0 if ok and ok_trampa else 1


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    sys.exit(asyncio.run(main(args.pregunta, args.trampa)))
