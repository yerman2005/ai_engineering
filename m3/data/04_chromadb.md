# Persistencia local con ChromaDB

## Por qué ChromaDB local

ChromaDB es una base de datos vectorial de código abierto pensada para ser simple. En modo local persistente no necesita Docker: guarda los datos en una carpeta del disco elegida al crear el cliente (`chromadb.PersistentClient(path="./vectorstore")`). Así el conocimiento del sistema sobrevive al reinicio del proceso y se puede actualizar sin reindexar todo.

Los datos se organizan en **colecciones**. Cada registro de una colección tiene un ID, el documento (texto), su embedding y metadatos opcionales.

## Operaciones CRUD

1. **Create (add / upsert)**: se guarda el texto junto con su vector. `upsert` es la mejor práctica: si el ID existe, lo actualiza; si no, lo crea.
2. **Read**:
   - `get()` busca por IDs exactos o por filtros de metadatos, como un SELECT tradicional.
   - `query()` busca por similitud semántica. Es el corazón de RAG.
3. **Update**: actualiza metadatos o el documento. Si cambia el texto, el vector debe recalcularse.
4. **Delete**: elimina vectores que ya no son relevantes.

## Metadatos

Los metadatos son etiquetas que se adjuntan a cada vector, por ejemplo `{"autor": "Juan", "fecha": "2024-05", "source": "pdf_v1"}`. Sirven para filtrar resultados antes de la búsqueda vectorial y para rastrear el origen de la información que usa el LLM, es decir, para citar fuentes.

## Integración con LangChain

El paquete `langchain-chroma` expone la clase `Chroma`, que recibe `persist_directory`, `collection_name` y una `embedding_function`. Con `add_documents` se indexan documentos y con `as_retriever(search_kwargs={"k": 4})` se obtiene un retriever que convierte la pregunta en embedding y devuelve los k fragmentos más parecidos. El retriever es un Runnable, así que se puede usar dentro de una cadena LCEL.

## Errores comunes

- **IDs no deterministas**: con IDs aleatorios es imposible actualizar de forma eficiente. Se recomiendan hashes del contenido o IDs estructurados (por ejemplo, `archivo-numero_de_chunk`).
- **Desajuste de embeddings**: consultar con un modelo de embeddings distinto del que se usó para indexar devuelve ruido aleatorio.
- **Ignorar la persistencia**: crear el cliente sin ruta de disco hace que los datos se pierdan al cerrar el script.
- **Reindexar siempre**: conviene verificar si la colección ya tiene datos antes de volver a indexar, para ahorrar tiempo y costo de embeddings.
- **Contexto infinito**: pasar 50 fragmentos al LLM causa errores de límite de tokens o el efecto "Lost in the Middle". Se recomienda un top_k de entre 3 y 5.
