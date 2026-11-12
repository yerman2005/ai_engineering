# Módulo 2 · Pipeline de procesamiento validado

Pre-entrega del Módulo 2 del curso **AI Engineering** (Coderhouse).

**Pipeline de Extracción de Entidades Técnicas** construido con LangChain (LCEL) y Pydantic. Recibe un párrafo de texto sin procesar, como un log de error o la descripción de una arquitectura, y devuelve un objeto validado con:

- las tecnologías mencionadas,
- el nivel de criticidad,
- un resumen técnico.

## Estructura

```
m2/
├── schemas.py         ← contrato de datos (modelo Pydantic)
├── chain.py           ← prompt + LLM + salida estructurada + reintentos + process_text()
├── main.py            ← mini-script de prueba asíncrono
├── requirements.txt
├── .env.example
└── README.md
```

## Componentes

- **Esquema Pydantic** (`schemas.py`). `ExtraccionTecnica` define tres campos:
  - `tecnologias`: lista de strings, no vacía. Se le quitan los espacios y los duplicados.
  - `nivel_de_criticidad`: enum `NivelDeCriticidad` con los valores `baja`, `media` y `alta`.
  - `resumen_tecnico`: string no vacío.
- **Prompt Template** (`chain.py`). Es un `ChatPromptTemplate` modular con dos variables:
  - `{texto}`: el texto de entrada.
  - `{instrucciones_formato}`: las instrucciones de formato, inyectadas con `.partial()`.

  No usa f-strings.
- **Cadena LCEL** (`chain.py`):

  ```python
  chain = (
      PROMPT
      | model.with_structured_output(ExtraccionTecnica, method="json_schema", include_raw=True)
      | RunnableLambda(_validar_salida)
  ).with_retry(...)
  ```

  El modelo es `ChatOpenAI` o `ChatAnthropic`. Se elige con `LLM_PROVIDER` y se configura igual que en el Módulo 1: las API keys se guardan como `SecretStr` y se pueden definir los reintentos del SDK y el timeout.
- **Resiliencia**:
  - `_validar_salida` revisa la respuesta cruda del modelo. Lanza `SalidaIncompletaError` en estos casos:
    - la respuesta se cortó por falta de tokens (`finish_reason == "length"` en OpenAI, `stop_reason == "max_tokens"` en Anthropic);
    - el JSON no se pudo parsear o no cumple el esquema.
  - `.with_retry()` reintenta la cadena ante `SalidaIncompletaError`, `ValidationError` u `OutputParserException`. Hace hasta 3 intentos, con backoff exponencial y jitter.
  - Los errores de red y de rate limit (429) los reintenta el SDK (`LLM_MAX_RETRIES`).
- **`process_text(text: str)`**: función asíncrona que ejecuta la cadena con `.ainvoke()`. Registra en los logs el inicio, cada validación fallida, cada reintento y el resultado final.

## Instalación (Python 3.12)

```bash
cd m2
py -3.12 -m venv .venv          # Windows   (Linux/macOS: python3.12 -m venv .venv)
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
python --version                # tiene que mostrar 3.12.x
pip install -r requirements.txt
```

## Configuración

Copiá `.env.example` como `.env` y completá los valores:

```bash
copy .env.example .env          # Linux/macOS: cp .env.example .env
```

| Variable | Descripción | Por defecto |
|---|---|---|
| `LLM_PROVIDER` | `openai` o `anthropic` | `openai` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | API key del proveedor elegido | — |
| `OPENAI_MODEL` | Modelo de OpenAI | `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | Modelo de Anthropic | `claude-sonnet-5-5` |
| `LLM_MAX_RETRIES` | Reintentos del SDK ante errores de red / 429 | `2` |
| `LLM_TIMEOUT_S` | Timeout por llamada, en segundos | `60` |

## Ejecución

```bash
python main.py
```

El script corre dos ejemplos:

1. Un log de error con FastAPI, Redis y PostgreSQL.
2. Una **prueba de estrés** con un texto ambiguo, para ver si el modelo se recupera o si el validador rechaza la salida.

Los logs muestran la validación y los posibles reintentos.

> El script hace llamadas reales a la API, y esas llamadas tienen costo.

Uso desde código:

```python
from chain import process_text

resultado = await process_text("FastAPI devuelve 503: el pool de PostgreSQL está agotado...")
print(resultado.model_dump_json(indent=2))
```

## Ejemplo de salida

Entrada:

```text
2026-11-12 10:41:07 ERROR api.orders - FastAPI devolvió 503 en /orders: TimeoutError al obtener
una conexión del pool de PostgreSQL (max_connections=100 alcanzado). La caché en Redis tiene un
hit rate del 12% y no alcanza a absorber la carga de 2.000 requests concurrentes en producción.
```

Salida validada:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```
