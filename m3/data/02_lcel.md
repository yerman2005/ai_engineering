# Orquestación con LangChain Expression Language (LCEL)

## Qué es LCEL

LCEL es un lenguaje declarativo para componer cadenas de procesamiento con LLMs. Se basa en una idea: cada componente es un bloque con una interfaz estándar, y los bloques se unen con el operador pipe `|`. La salida del componente de la izquierda es la entrada del de la derecha.

Al usar LCEL se obtiene, sin código extra, soporte asíncrono, streaming automático, ejecución en paralelo y observabilidad con herramientas como LangSmith.

## El protocolo Runnable

Casi todos los objetos de LangChain (modelos, prompts, parsers, retrievers, herramientas) implementan el protocolo Runnable, que expone siempre los mismos métodos:

- `invoke()`: llama al componente con una entrada (síncrono).
- `ainvoke()`: versión asíncrona de `invoke`.
- `batch()` / `abatch()`: procesa una lista de entradas optimizando la latencia.
- `stream()` / `astream()`: devuelve fragmentos de la respuesta a medida que se generan.

## Estructura típica de una cadena

El flujo estándar es `PromptTemplate | ChatModel | OutputParser`:

1. El `ChatPromptTemplate` recibe un diccionario de variables y genera un `ChatPromptValue`.
2. El modelo recibe ese valor, lo envía al LLM y devuelve un `AIMessage`.
3. El parser extrae el contenido y devuelve un string (`StrOutputParser`) o un objeto validado (`PydanticOutputParser`).

`ChatPromptTemplate.from_messages` permite definir roles (system, human, ai), clave para que el modelo siga instrucciones de seguridad o de formato.

## Modelado de entradas y paralelismo

Con `RunnablePassthrough` o con funciones lambda se puede "modelar" la entrada dentro de la cadena. Por ejemplo, un diccionario `{"context": retriever, "question": RunnablePassthrough()}` arma las dos variables que espera el prompt a partir de una sola pregunta. LCEL convierte automáticamente diccionarios y funciones en Runnables.

`RunnableParallel` ejecuta varias sub-cadenas al mismo tiempo sobre la misma entrada y devuelve un diccionario con los resultados. El tiempo total es el de la rama más lenta, no la suma.

## Salidas estructuradas y resiliencia

Los LLMs son probabilísticos y su salida natural es texto libre. Para tener sistemas predecibles se define un contrato de datos con Pydantic y se usa `model.with_structured_output(Esquema)` o un `PydanticOutputParser`. Pydantic valida la sintaxis (JSON parseable), la estructura (campos requeridos) y la semántica (reglas de negocio, por ejemplo un valor de confianza entre 0 y 1).

Para la resiliencia se envuelve la cadena con `.with_retry()`, que reintenta con backoff exponencial ante errores transitorios, y con `.with_fallbacks()` para redirigir la carga a otro modelo si el principal falla. No se deben reintentar errores permanentes, como un prompt con un error lógico.

## Errores comunes

- Pasar un string a una cadena cuyo prompt espera un diccionario con variables.
- No terminar la cadena con un parser y recibir un `AIMessage` en lugar de texto.
- Usar métodos síncronos dentro de una aplicación web: conviene usar siempre `ainvoke`, `abatch` y `astream`.
