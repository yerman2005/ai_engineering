"""Mini-script de prueba: ejecuta el pipeline sobre un ejemplo y sobre un texto ambiguo.

Uso:
    python main.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

from chain import process_text

EJEMPLO = (
    "2026-11-12 10:41:07 ERROR api.orders - FastAPI devolvió 503 en /orders: "
    "TimeoutError al obtener una conexión del pool de PostgreSQL (max_connections=100 "
    "alcanzado). La caché en Redis tiene un hit rate del 12% y no alcanza a absorber "
    "la carga de 2.000 requests concurrentes en producción."
)

# Prueba de estrés: texto ambiguo, sin tecnologías explícitas.
EJEMPLO_AMBIGUO = "Ayer algo andaba lento, pero hoy parece que funciona. No sé bien qué pasó."


async def ejecutar(titulo: str, texto: str) -> bool:
    print(f"\n{'=' * 60}\n{titulo}\n{'=' * 60}\nEntrada: {texto}\n")
    try:
        resultado = await process_text(texto)
    except Exception as exc:  # se muestra el error en lugar de cortar el script
        print(f"[error] {type(exc).__name__}: {exc}")
        return False
    print(resultado.model_dump_json(indent=2))
    return True


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Silencia los logs de HTTP para que se vean los del pipeline.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    ok = await ejecutar("Ejemplo: log de error", EJEMPLO)
    await ejecutar("Prueba de estrés: texto ambiguo", EJEMPLO_AMBIGUO)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
