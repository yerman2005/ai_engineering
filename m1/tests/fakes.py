"""Dobles de prueba compartidos: reemplazan a los SDKs para no hacer llamadas reales."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from types import SimpleNamespace
from typing import Any

import httpx2

from llm_client import BaseLLMClient, ChatMessage, GenerationConfig, LLMError, ModelResponse

_REQUEST = httpx2.Request("POST", "https://api.example.test/v1")


def sdk_status_error(error_cls: type[Exception], status_code: int) -> Exception:
    """Instancia una excepción de estado HTTP de `openai` o `anthropic`."""
    response = httpx2.Response(status_code, request=_REQUEST)
    return error_cls(f"HTTP {status_code}", response=response, body=None)


def sdk_request_error(error_cls: type[Exception]) -> Exception:
    """Instancia `APIConnectionError` o `APITimeoutError` de un SDK."""
    return error_cls(request=_REQUEST)


async def slow(seconds: float, value: Any = None) -> Any:
    await asyncio.sleep(seconds)
    return value


class FakeOpenAIStream:
    """Imita `openai.AsyncStream`: iterable asíncrono y context manager."""

    def __init__(
        self, chunks: Iterable[Any], *, delay: float = 0.0, error: Exception | None = None
    ):
        self._chunks = list(chunks)
        self._delay = delay
        self._error = error
        self.closed = False

    async def __aenter__(self) -> FakeOpenAIStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True

    async def __aiter__(self) -> AsyncIterator[Any]:
        for chunk in self._chunks:
            await asyncio.sleep(self._delay)
            yield chunk
        if self._error is not None:
            raise self._error


def openai_chunk(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class FakeAnthropicStream:
    """Imita el context manager que devuelve `AsyncAnthropic.messages.stream()`."""

    def __init__(self, texts: Iterable[str], *, error: Exception | None = None):
        self._texts = list(texts)
        self._error = error
        self.closed = False

    async def __aenter__(self) -> FakeAnthropicStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True

    @property
    async def text_stream(self) -> AsyncIterator[str]:
        for text in self._texts:
            yield text
        if self._error is not None:
            raise self._error


class FakeLLMClient(BaseLLMClient):
    """Cliente en memoria para probar el manager sin depender de ningún SDK."""

    provider_name = "fake"
    sdk_error_types = ()

    def __init__(
        self,
        *,
        reply: str = "respuesta",
        chunks: Sequence[str] = ("hola", " ", "mundo"),
        error: LLMError | None = None,
        delay: float = 0.0,
    ) -> None:
        super().__init__(default_model="fake-model")
        self.reply = reply
        self.chunks = chunks
        self.error = error
        self.delay = delay
        self.calls: list[list[ChatMessage]] = []
        self.active = 0
        self.max_active = 0
        self.closed = False

    async def _generate(
        self, messages: Sequence[ChatMessage], model: str, config: GenerationConfig
    ) -> ModelResponse:
        self.calls.append(list(messages))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.error is not None:
                raise self.error
            content = f"{self.reply}:{messages[-1].content}"
            return ModelResponse(content=content, provider=self.provider_name, model=model)
        finally:
            self.active -= 1

    async def _stream(
        self, messages: Sequence[ChatMessage], model: str, config: GenerationConfig
    ) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error

    def _map_error(self, exc: Exception) -> LLMError:  # pragma: no cover - sin errores de SDK
        raise NotImplementedError

    async def aclose(self) -> None:
        self.closed = True
