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
from collections.abc import Callable

from ingest import ingestar
from rag import get_rag_response
from schemas import RespuestaRAG

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


RESPUESTA_DESCONOCIDA = "no lo sé"


def es_no_lo_se(resultado: RespuestaRAG) -> bool:
    """True si el modelo declaró no saber (ignora mayúsculas y puntuación final)."""
    return resultado.respuesta.strip().rstrip(".!").lower() == RESPUESTA_DESCONOCIDA


def verificar_respondible(resultado: RespuestaRAG) -> str | None:
    """La respuesta debe salir de los documentos: no 'No lo sé' y con al menos una referencia."""
    if es_no_lo_se(resultado):
        return "el modelo respondió 'No lo sé' aunque la respuesta está en los documentos"
    if not resultado.referencias:
        return "la respuesta no cita ninguna referencia"
    return None


def verificar_trampa(resultado: RespuestaRAG) -> str | None:
    """Sin alucinación: debe responder exactamente 'No lo sé' y sin referencias."""
    if not es_no_lo_se(resultado):
        return f"posible alucinación: se esperaba 'No lo sé' y respondió {resultado.respuesta!r}"
    if resultado.referencias:
        return f"se esperaban referencias vacías y devolvió {resultado.referencias}"
    return None


async def ejecutar(
    titulo: str,
    pregunta: str,
    verificar: Callable[[RespuestaRAG], str | None],
) -> bool:
    print(f"\n{'=' * 60}\n{titulo}\n{'=' * 60}\nPregunta: {pregunta}\n")
    try:
        resultado = await get_rag_response(pregunta)
    except Exception as exc:  # se muestra el error en lugar de cortar el script
        print(f"[error] {type(exc).__name__}: {exc}")
        return False
    print(resultado.model_dump_json(indent=2))

    error = verificar(resultado)
    print(f"\n[FALLO] {error}" if error else "\n[OK] verificación superada")
    return error is None


async def main(pregunta: str = PREGUNTA, pregunta_trampa: str = PREGUNTA_TRAMPA) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Silencia los logs de HTTP para que se vean los del sistema RAG.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    await asyncio.to_thread(ingestar)
    ok = await ejecutar(
        "Prueba 1: pregunta respondible con los documentos", pregunta, verificar_respondible
    )
    ok_trampa = await ejecutar(
        "Prueba 2: pregunta trampa (fuera de contexto)", pregunta_trampa, verificar_trampa
    )

    print(f"\n{'=' * 60}")
    print(f"Prueba 1 (respondible): {'OK' if ok else 'FALLO'}")
    print(f"Prueba 2 (trampa, sin alucinar): {'OK' if ok_trampa else 'FALLO'}")
    return 0 if ok and ok_trampa else 1


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    sys.exit(asyncio.run(main(args.pregunta, args.trampa)))
