# Arquitectura asíncrona con Python 3.12

## Por qué asyncio en ingeniería de IA

En un sistema de IA el tiempo casi no se gasta calculando: se gasta esperando la red. Esperamos a que el prompt llegue al proveedor, a que el modelo procese y a que los tokens vuelvan. Asyncio permite que un solo proceso gestione muchas tareas de entrada/salida (I/O) de forma concurrente. No es paralelismo real sino concurrencia cooperativa: mientras una tarea espera datos de la red, cede el control para que otra avance.

Si un agente debe buscar en una base vectorial, consultar una API de clima y generar una respuesta con un LLM, y cada paso tarda 2 segundos, el modelo síncrono tarda 6 segundos. Con asyncio las dos primeras consultas pueden correr al mismo tiempo.

## Conceptos fundamentales

- **Corrutina** (`async def` / `await`): una función que puede pausar su ejecución en cada `await` y liberar el hilo para otras tareas.
- **Event loop**: el motor que coordina las corrutinas. Es como el gestor de un restaurante que, si un camarero espera a la cocina, lo manda a atender otra mesa.
- **Task**: una corrutina programada para ejecutarse lo antes posible en el event loop, en segundo plano.

## Patrones de concurrencia

El punto de entrada recomendado es `asyncio.run(main())`: crea el loop, espera a que terminen las tareas y cierra los recursos de forma limpia. Desde Python 3.12 no se debe usar `asyncio.get_event_loop()` de forma descuidada; dentro de una función conviene `asyncio.get_running_loop()`.

`asyncio.gather` dispara varias corrutinas a la vez y espera a todas. Con `return_exceptions=True`, si una falla, el lote no se cae: la excepción vuelve como un resultado más.

`asyncio.TaskGroup` (Python 3.11+) gestiona grupos de tareas de forma más segura que `gather`: si una falla, las demás se cancelan limpiamente.

## Timeouts y semáforos

Nunca se debe dejar una llamada a un LLM abierta indefinidamente. `async with asyncio.timeout(10.0):` corta la espera y lanza `TimeoutError`, que se captura para aplicar un fallback.

Si se disparan 1.000 llamadas a la vez se recibe un error `429: Too Many Requests`. Un `asyncio.Semaphore(5)` limita la cantidad de tareas concurrentes activas: todas se encolan, pero solo 5 corren en simultáneo. Es el patrón correcto para generar embeddings de 10.000 filas.

## Anti-patrones

- Asyncio no acelera el cálculo. Para tareas CPU-bound (Pandas pesado, inferencia local de PyTorch) se usa `multiprocessing`.
- Bloquear el loop: llamar a `time.sleep(5)` dentro de una función `async` congela a todos los usuarios. Hay que usar `await asyncio.sleep(5)`. Si una librería es síncrona, se envuelve con `await asyncio.to_thread(funcion, datos)`.

## Streaming de tokens

Con generadores asíncronos se procesa cada token a medida que llega de la red y se envía al frontend por WebSockets o Server-Sent Events. Esto reduce el Time to First Token (TTFT), una de las métricas de experiencia de usuario más importantes en productos de IA.
