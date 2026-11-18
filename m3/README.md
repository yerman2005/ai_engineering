# Módulo 3 · Sistema de recuperación semántica local (RAG)

Pre-entrega del Módulo 3 del curso **AI Engineering** (Coderhouse).

Flujo **RAG end-to-end** con ChromaDB local, LangChain (LCEL) y Pydantic:

1. ingiere los documentos de `data/`, los fragmenta y los persiste en ChromaDB;
2. convierte la pregunta del usuario en un embedding y recupera los fragmentos más relevantes;
3. genera una respuesta que usa **solo** esa información. Si la respuesta no está en el contexto, contesta "No lo sé".

## Estructura

```
m3/
├── data/              ← dataset de ejemplo (4 apuntes .md del curso)
├── config.py          ← configuración del LLM (misma lógica que M1/M2), embeddings y ChromaDB
├── schemas.py         ← contrato de salida (modelo Pydantic RespuestaRAG)
├── ingest.py          ← ingesta: carga + chunking + persistencia en ChromaDB
├── rag.py             ← retriever + cadena LCEL + get_rag_response()
├── main.py            ← script de prueba (pregunta respondible + pregunta trampa)
├── requirements.txt
├── .env.example
└── README.md
```

La base vectorial es la instalación local de ChromaDB en `../chromadb-local`. La colección `m3_rag` se guarda en `../chromadb-local/vectorstore`.

## Componentes

- **Ingesta** (`ingest.py`). `ingestar()` hace lo siguiente:
  - Si la colección ya tiene datos, no reindexa. Así no se repite el costo de los embeddings.
  - Lee los archivos `.txt` y `.md` de `data/`.
  - Los fragmenta con `RecursiveCharacterTextSplitter.from_tiktoken_encoder` en chunks de **500 tokens con 50 de overlap**, medidos con `tiktoken`.
  - Les agrega el metadato `source` (nombre del archivo) y un ID determinista (`<archivo>-<n>`).
  - Los persiste con `langchain_chroma.Chroma` (`persist_directory`).
- **Embeddings**: OpenAI `text-embedding-3-small`. Se usa **el mismo modelo** para indexar y para consultar.
- **Recuperación** (`rag.py`): `vectorstore.as_retriever(search_kwargs={"k": 4})`. Convierte la pregunta en un embedding y devuelve los 4 fragmentos más similares (top_k = 4).
- **Generación grounded** (`rag.py`). Es una cadena LCEL:

  ```python
  chain = (
      {"contexto": retriever | RunnableLambda(formatear_docs), "pregunta": RunnablePassthrough()}
      | PROMPT
      | model
      | PydanticOutputParser(pydantic_object=RespuestaRAG)
  )
  ```

  - El prompt de sistema funciona como "filtro de veracidad". Indica responder solo con el CONTEXTO y, si la respuesta no está ahí, decir "No lo sé" y dejar las referencias vacías.
  - Cada fragmento llega al modelo con la etiqueta `[fuente: archivo]`, para que pueda citar las fuentes.
  - El modelo es `ChatOpenAI` o `ChatAnthropic`, según `LLM_PROVIDER`.
- **Salida validada** (`schemas.py`). `RespuestaRAG` tiene dos campos:
  - `respuesta`: string no vacío.
  - `referencias`: lista con los archivos fuente usados.
- **`get_rag_response(query: str)`**: función asíncrona que ejecuta la cadena con `.ainvoke()`. Registra en los logs la consulta, los fragmentos recuperados y la respuesta validada.

## Instalación

m3 se ejecuta con el entorno virtual de `chromadb-local`, que ya tiene ChromaDB instalado:

```bash
cd m3
..\chromadb-local\.venv\Scripts\activate
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
| `OPENAI_API_KEY` | **Obligatoria siempre** (embeddings). También es la key del LLM si el proveedor es `openai`. | — |
| `ANTHROPIC_API_KEY` | API key de Anthropic (solo si `LLM_PROVIDER=anthropic`) | — |
| `OPENAI_MODEL` | Modelo de OpenAI | `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | Modelo de Anthropic | `claude-sonnet-5-5` |
| `LLM_MAX_RETRIES` | Reintentos del SDK ante errores de red / 429 | `2` |
| `LLM_TIMEOUT_S` | Timeout por llamada, en segundos | `60` |

## Ejecución

```bash
python ingest.py      # opcional: indexa data/ (si la colección ya existe, no hace nada)
python main.py        # ingesta si hace falta + las dos pruebas
```

`main.py` corre dos pruebas:

1. **Pregunta respondible**: "¿Para qué sirve un asyncio.Semaphore cuando se llama a un LLM?". La respuesta está en `data/01_asyncio.md`.
2. **Pregunta trampa**: "¿Cuál es la capital de Australia?". No está en los documentos, así que el modelo tiene que responder "No lo sé" sin alucinar.

Para reindexar desde cero, borrá la carpeta `../chromadb-local/vectorstore`.

> El script hace llamadas reales a las APIs (embeddings y LLM), y esas llamadas tienen costo.

Uso desde código:

```python
from rag import get_rag_response

resultado = await get_rag_response("¿Qué es el overlap en el chunking?")
print(resultado.model_dump_json(indent=2))
```

## Ejemplo de salida

Prueba 1 (pregunta respondible):

```json
{
  "respuesta": "Un asyncio.Semaphore limita la cantidad de llamadas concurrentes activas: todas las tareas se encolan, pero solo N corren en simultáneo, lo que evita errores 429 (Too Many Requests).",
  "referencias": ["01_asyncio.md"]
}
```

Prueba 2 (pregunta trampa):

```json
{
  "respuesta": "No lo sé",
  "referencias": []
}
```
