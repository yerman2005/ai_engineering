"""Script de validación: hace la misma pregunta en modo normal y en streaming.

Uso:
    python main.py                       # proveedor definido en LLM_PROVIDER
    python main.py --provider anthropic  # un proveedor puntual
    python main.py --provider all        # todos los que tengan API key
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from llm_client import (
    SUPPORTED_PROVIDERS,
    AsyncLLMManager,
    LLMError,
    ProviderName,
    Settings,
)

DEFAULT_QUESTION = "¿Qué es la entropía?"
SYSTEM_PROMPT = "Respondé en español, de forma clara y en no más de tres párrafos."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba del Unified Async LLM Client.")
    parser.add_argument(
        "--provider",
        choices=[*SUPPORTED_PROVIDERS, "all"],
        default=None,
        help="Proveedor a usar. Por defecto, el de la variable LLM_PROVIDER.",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Pregunta a enviar.")
    return parser.parse_args(argv)


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


async def run_provider(settings: Settings, provider: ProviderName, question: str) -> bool:
    """Ejecuta la prueba completa para un proveedor. Devuelve True si todo salió bien."""
    print_header(f"Proveedor: {provider}")
    try:
        manager = AsyncLLMManager.from_settings(settings, provider)
    except LLMError as exc:
        print(f"[error de configuración] {exc}")
        return False

    success = True
    async with manager:
        # 1) Modo normal: el manager devuelve un resultado, nunca lanza.
        print(f"\n--- Modo normal ---\nPregunta: {question}\n")
        result = await manager.generate(question, system=SYSTEM_PROMPT)
        if result.response is not None:
            response = result.response
            print(response.content)
            tokens = response.usage.total_tokens if response.usage else "n/d"
            print(
                f"\n(modelo: {response.model} | tokens: {tokens} | fin: {response.finish_reason})"
            )
        elif result.error is not None:
            print(f"[error] {result.error.code}: {result.error.message}")
            success = False

        # 2) Streaming: los fragmentos se imprimen a medida que llegan.
        print("\n--- Modo streaming ---\n")
        try:
            async for chunk in manager.stream(question, system=SYSTEM_PROMPT):
                print(chunk, end="", flush=True)
            print()
        except LLMError as exc:
            print(f"\n[error en streaming] {exc}")
            success = False

    return success


async def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    try:
        settings = Settings.from_env()
    except ValidationError as exc:
        print(f"Configuración inválida en el entorno o en .env:\n{exc}", file=sys.stderr)
        return 2

    providers: list[ProviderName]
    if args.provider == "all":
        providers = settings.configured_providers()
        if not providers:
            print(
                "No hay API keys configuradas. Definí OPENAI_API_KEY o ANTHROPIC_API_KEY "
                "(ver .env.example).",
                file=sys.stderr,
            )
            return 2
    else:
        providers = [args.provider or settings.provider]

    # Los proveedores se prueban uno después del otro para que la salida se lea ordenada.
    results = [await run_provider(settings, p, args.question) for p in providers]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
