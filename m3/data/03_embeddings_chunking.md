# Embeddings, similitud y chunking

## Qué es un embedding

Un embedding es una representación numérica densa de un texto en un espacio vectorial de alta dimensión: una lista de números que captura su significado. El modelo `text-embedding-3-small` de OpenAI devuelve un vector de 1536 dimensiones.

Antes se usaba one-hot encoding: cada palabra era una posición en un vector gigante de ceros con un solo 1. Con esa representación la distancia entre "perro" y "canino" era la misma que entre "perro" y "microprocesador". Los embeddings densos, en cambio, ubican cerca los conceptos relacionados.

Los modelos estáticos (Word2Vec, GloVe) asignaban un único vector por palabra. Los modelos contextuales (BERT, Ada, Voyage) generan vectores distintos según el contexto: distinguen "banco" de plaza y "banco" financiero.

## Métricas de similitud

- **Similitud coseno**: mide el ángulo entre dos vectores. Ignora la longitud del texto y se enfoca en el tema. Es el estándar en RAG. Su rango va de -1 a 1.
- **Distancia euclídea (L2)**: distancia en línea recta. Es sensible a la magnitud; se usa cuando los vectores están normalizados.
- **Producto punto**: combina magnitud y ángulo. Es común en sistemas de recomendación.

## Errores comunes con embeddings

1. Tratarlos como comprensión total: "hay stock" y "no hay stock" pueden quedar cerca.
2. Ignorar el límite de tokens del modelo de embeddings: un documento muy largo se trunca o pierde precisión.
3. Mezclar modelos: nunca se compara un embedding de OpenAI con uno de un modelo local de HuggingFace. Viven en espacios vectoriales distintos.
4. Ignorar el domain shift: un modelo general puede fallar en lenguaje médico o código muy técnico.

## Por qué fragmentar (chunking)

Si se convierte un PDF de 50 páginas en un solo vector, se obtiene un promedio semántico tan diluido que no sirve para encontrar respuestas específicas. El chunking crea unidades de información atómicas y semánticamente coherentes.

Antes de dividir hay que limpiar: encabezados y pies de página repetidos, saltos de línea múltiples, caracteres basura de tablas mal extraídas.

## Estrategias

- **Tamaño fijo**: corta cada X caracteres o tokens. Es simple pero puede partir una oración a la mitad.
- **Recursive Character Splitting**: el estándar en LangChain (`RecursiveCharacterTextSplitter`). Intenta dividir primero por párrafos (`\n\n`), luego por saltos de línea (`\n`), luego por espacios y, como último recurso, por caracteres.
- **Overlap (solapamiento)**: el final de un chunk se repite al inicio del siguiente para que un concepto ubicado en el límite no se pierda.

Conviene medir los chunks en tokens con un tokenizador real como `tiktoken`, porque los modelos limitan y cobran por tokens, no por caracteres. Un buen chunking equilibra granularidad, contexto y limpieza.
