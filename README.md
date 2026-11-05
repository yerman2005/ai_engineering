# AI Engineering · Coderhouse

Repositorio con las entregas del curso **AI Engineering** de Coderhouse. Cada módulo del curso tiene su propia carpeta, con un proyecto independiente: su código, sus dependencias, su configuración y su propio README.

## Contenido

| Carpeta | Módulo | Entrega | Estado |
|---|---|---|---|
| [`m1/`](m1/) | Módulo 1 · La interfaz base: conexión y abstracción de LLMs | Pre-entrega: **Cliente de LLM robusto y asíncrono** | Entregado |

A medida que avance el curso se irán sumando las carpetas de los módulos siguientes (`m2/`, `m3/`, …).

## `m1/` · Pre-entrega del Módulo 1

`m1` es la **pre-entrega del Módulo 1**. Es una capa de abstracción asíncrona en Python 3.12 para trabajar con **OpenAI** y **Anthropic** a través de una interfaz común (`AsyncLLMManager`):

- **Intercambiabilidad**: el proveedor se elige por configuración (`LLM_PROVIDER`), sin tocar la lógica de negocio.
- **Asincronía**: usa los SDKs oficiales asíncronos (`AsyncOpenAI` / `AsyncAnthropic`).
- **Streaming**: devuelve la respuesta fragmento a fragmento con un generador asíncrono.
- **Validación**: mensajes, parámetros y configuración validados con Pydantic; las API keys se guardan como `SecretStr`.
- **Resiliencia**: timeouts, reintentos con backoff y excepciones propias.
- **Tests**: suite de `pytest` con clientes falsos, sin llamadas reales a las APIs.

Para instalarlo, configurarlo y ejecutarlo, ver [`m1/README.md`](m1/README.md).

### Inicio rápido

```bash
cd m1
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
copy .env.example .env          # Linux/macOS: cp .env.example .env  → completar las API keys
python main.py
pytest
```

## Estructura del repositorio

```
ai_engineering/
├── README.md          ← este archivo
└── m1/                ← pre-entrega del Módulo 1
    ├── llm_client/    ← paquete con la abstracción de clientes
    ├── tests/         ← tests con pytest
    ├── main.py        ← demo de uso
    ├── pyproject.toml
    ├── .env.example
    └── README.md      ← documentación detallada del módulo
```

## Requisitos generales

- **Python 3.12** o superior.
- Las API keys de los proveedores que se usen en cada módulo (por ejemplo, OpenAI y/o Anthropic).
- Se recomienda un entorno virtual separado para cada módulo, ya que cada uno tiene sus propias dependencias.

## Convenciones

- Cada módulo es autocontenido: se instala y se ejecuta desde su propia carpeta.
- Los secretos van en un archivo `.env` local, que **no se versiona**. Cada módulo incluye un `.env.example` como plantilla.
- No se versionan entornos virtuales, `__pycache__` ni configuraciones del editor (`.vscode`).

## Autor

Germán Schmidt
