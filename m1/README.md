# Unified Async LLM Client

Pre-entrega del **Módulo 1 · La interfaz base: conexión y abstracción de LLMs** (AI Engineering, Coderhouse).

Capa de abstracción asíncrona para trabajar con **OpenAI** y **Anthropic** a través de una interfaz común:

- **Intercambiabilidad**: el proveedor se elige por configuración (`LLM_PROVIDER`). La lógica de negocio usa siempre `AsyncLLMManager`, sin importar el proveedor.
- **Asincronía**: todas las llamadas usan los clientes oficiales asíncronos (`AsyncOpenAI` / `AsyncAnthropic`), así que nunca bloquean el event loop.
- **Streaming**: `stream()` es un generador asíncrono que devuelve los fragmentos de texto a medida que llegan.
- **Validación**: los mensajes, los parámetros del modelo y la configuración se validan con Pydantic. Las API keys se guardan como `SecretStr`.
- **Resiliencia**: hay timeouts, reintentos con backoff y errores traducidos a excepciones propias. En modo normal, un fallo se devuelve como un resultado estructurado y el programa sigue.

## Requisitos

- Python **3.12 o superior**
- Una API key de OpenAI y/o de Anthropic

## Instalación

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e ".[dev]"     # dependencias + herramientas de desarrollo (pytest, ruff)
```

`pip install -e .` instala solo las dependencias de ejecución: `openai`, `anthropic`, `pydantic` y `python-dotenv`.

## Configuración

Copiá `.env.example` como `.env` y completá las variables:

| Variable | Obligatoria | Descripción | Por defecto |
|---|---|---|---|
| `LLM_PROVIDER` | No | Proveedor por defecto: `openai` o `anthropic` | `openai` |
| `OPENAI_API_KEY` | Si usás OpenAI | API key de OpenAI | — |
| `ANTHROPIC_API_KEY` | Si usás Anthropic | API key de Anthropic | — |
| `OPENAI_MODEL` | No | Modelo de OpenAI | `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | No | Modelo de Anthropic | `claude-sonnet-5-5` |
| `LLM_MAX_RETRIES` | No | Reintentos automáticos ante 429, 5xx y errores de red (0 a 10) | `2` |
| `LLM_TIMEOUT_S` | No | Timeout por llamada en segundos. En streaming es la espera máxima entre fragmentos | `60` |

Las variables definidas en el entorno tienen prioridad sobre las de `.env`. El archivo `.env` está en `.gitignore`.

## Ejecución del script de prueba

`main.py` carga el `.env` y pregunta *"¿Qué es la entropía?"*, primero en **modo normal** y después en **streaming**:

```bash
python main.py                         # usa el proveedor de LLM_PROVIDER
python main.py --provider anthropic    # un proveedor puntual
python main.py --provider all          # todos los proveedores con API key configurada
python main.py --question "¿Qué es un LLM?"
```

> El script hace llamadas reales a las APIs, y esas llamadas tienen costo.

Códigos de salida: `0` si todo salió bien, `1` si alguna llamada falló, `2` si la configuración es inválida.

## Uso como librería

```python
import asyncio

from llm_client import AsyncLLMManager, GenerationConfig, Settings


async def main() -> None:
    settings = Settings.from_env()
    async with AsyncLLMManager.from_settings(settings) as llm:
        # Modo normal: nunca lanza; devuelve una respuesta o un error estructurado.
        result = await llm.generate(
            "¿Qué es la entropía?",
            system="Respondé en una oración.",
            config=GenerationConfig(temperature=0.2, max_tokens=300),
        )
        if result.ok:
            print(result.response.content)
        else:
            print(result.error.code, result.error.message)

        # Streaming: generador asíncrono de fragmentos de texto.
        async for chunk in llm.stream("Contame un dato curioso"):
            print(chunk, end="", flush=True)

        # Lote con concurrencia limitada (semáforo) para no provocar errores 429.
        results = await llm.generate_many(["Hola", "¿Qué es un token?"], max_concurrency=2)


asyncio.run(main())
```

## Arquitectura

```
llm_client/
├── schemas.py           # Pydantic: ChatMessage, GenerationConfig, ModelResponse, LLMResult, ErrorInfo
├── config.py            # Settings: configuración desde entorno/.env, API keys como SecretStr
├── exceptions.py        # LLMError y subclases (autenticación, rate limit, red, timeout...)
├── _errors.py           # traducción de excepciones de los SDKs a LLMError
├── base.py              # BaseLLMClient (ABC): contrato común, timeouts y manejo de errores
├── openai_client.py     # OpenAIClient    (AsyncOpenAI, Chat Completions)
├── anthropic_client.py  # AnthropicClient (AsyncAnthropic, Messages API)
└── manager.py           # AsyncLLMManager: factory por configuración + fachada para el negocio
main.py                  # script de validación (modo normal + streaming)
tests/                   # tests con pytest; los SDKs se reemplazan por dobles (sin red)
```

### Decisiones de diseño

- **Inversión de dependencias**: el código de negocio depende de `AsyncLLMManager` y `BaseLLMClient`, nunca de `openai` ni de `anthropic`. Las excepciones de los SDKs se traducen a `LLMError` dentro de cada cliente.
- **Factory**: `AsyncLLMManager.from_settings()` decide qué cliente concreto crear a partir de `LLM_PROVIDER`.
- **Template method**: `BaseLLMClient.generate()` y `stream()` aplican el timeout (`asyncio.timeout`) y la traducción de errores para todos los proveedores. Cada subclase implementa solo la llamada a su SDK.
- **Diferencias que se abstraen**:
  - OpenAI devuelve el texto en `choices[0].message.content`.
  - Anthropic devuelve una lista de bloques: se concatenan solo los de tipo `text`, porque puede haber bloques `thinking` antes.
  - En Anthropic, el prompt de sistema va en el parámetro `system`.
- **Errores en modo normal vs. streaming**:
  - `generate()` devuelve un `LLMResult` con `response` o con `error`, así que el flujo nunca se interrumpe.
  - En `stream()` el fallo puede llegar después de algunos fragmentos ya entregados, así que se lanza un `LLMError` controlado (nunca una excepción del SDK) y quien consume decide qué hacer.
- **Reintentos**: se usan los de los SDKs oficiales (`max_retries`). Reintentan con backoff exponencial los errores 408, 409, 429, 5xx y de conexión, así que no hace falta reimplementarlos.
- **Concurrencia controlada**: `generate_many()` combina `asyncio.Semaphore` con `asyncio.gather` para limitar las llamadas simultáneas.
- **`temperature` en Anthropic**: la versión actual de la API de Claude ya no admite parámetros de muestreo. El esquema valida el rango 0 a 2 para todos los proveedores, pero `AnthropicClient` no envía el valor y deja una advertencia en el log.

### Agregar un proveedor

1. Crear una subclase de `BaseLLMClient` que implemente `_generate`, `_stream`, `_map_error` y `aclose`.
2. Registrarla en `_PROVIDERS` (`manager.py`) y agregar su nombre y API key a `Settings`.

También se puede pasar cualquier `BaseLLMClient` directamente a `AsyncLLMManager(client)`.

## Calidad de código

```bash
pytest                    # suite completa (no hace llamadas reales ni gasta créditos)
pytest tests/test_manager.py::test_generate_devuelve_error_controlado_en_lugar_de_lanzar   # un solo test
ruff check .              # lint
ruff format .             # formato
```
