"""Recuperación + generación grounded con LangChain (LCEL).

Cadena:
    {"contexto": retriever | formatear_docs, "pregunta": RunnablePassthrough()}
    | prompt | model | PydanticOutputParser(RespuestaRAG)
"""

from __future__ import annotations

import logging
from functools import cache

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from config import TOP_K, Settings, build_model
from ingest import get_vectorstore
from schemas import RespuestaRAG

logger = logging.getLogger(__name__)

PARSER = PydanticOutputParser(pydantic_object=RespuestaRAG)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un asistente técnico. Respondé SOLO basándote en el CONTEXTO proporcionado. "
            "No uses conocimiento propio ni inventes datos. Si la respuesta no está en el "
            "CONTEXTO, respondé exactamente \"No lo sé\" y dejá las referencias vacías.\n"
            "En 'referencias' incluí los nombres de archivo indicados como [fuente: ...] "
            "de los fragmentos que usaste.\n\n"
            "{format_instructions}",
        ),
        ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ]
).partial(format_instructions=PARSER.get_format_instructions())


def formatear_docs(docs: list[Document]) -> str:
    """Une los fragmentos recuperados indicando la fuente de cada uno."""
    for doc in docs:
        logger.info("Recuperado: %s", doc.metadata.get("source"))
    return "\n\n".join(f"[fuente: {d.metadata.get('source')}]\n{d.page_content}" for d in docs)


@cache
def get_chain() -> Runnable[str, RespuestaRAG]:
    """Construye (una sola vez) la cadena RAG a partir de la configuración del entorno."""
    settings = Settings.from_env()
    logger.info("Proveedor configurado: %s", settings.provider)
    retriever = get_vectorstore(settings).as_retriever(search_kwargs={"k": TOP_K})
    return (
        {"contexto": retriever | RunnableLambda(formatear_docs), "pregunta": RunnablePassthrough()}
        | PROMPT
        | build_model(settings)
        | PARSER
    )


async def get_rag_response(query: str) -> RespuestaRAG:
    """Busca en ChromaDB, arma el prompt con los fragmentos y devuelve la respuesta validada."""
    if not query or not query.strip():
        raise ValueError("La consulta no puede estar vacía.")

    logger.info("Consulta: %s", query)
    resultado = await get_chain().ainvoke(query)
    logger.info("Respuesta validada: %s", resultado.model_dump(mode="json"))
    return resultado
