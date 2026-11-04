from types import SimpleNamespace
from unittest.mock import AsyncMock

import openai
import pytest
from pydantic import SecretStr

from llm_client import (
    ChatMessage,
    GenerationConfig,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    OpenAIClient,
    Role,
)
from tests.fakes import FakeOpenAIStream, openai_chunk, sdk_request_error, sdk_status_error, slow

MESSAGES = [
    ChatMessage(role=Role.SYSTEM, content="Sé breve."),
    ChatMessage(role=Role.USER, content="¿Qué es la entropía?"),
]


def make_sdk(create: AsyncMock) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=AsyncMock(),
    )


def make_completion(text: str = "Una medida del desorden.") -> SimpleNamespace:
    return SimpleNamespace(
        model="gpt-4o-mini-2024",
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=6),
    )


def make_client(create: AsyncMock) -> OpenAIClient:
    return OpenAIClient(api_key=SecretStr("sk-test"), sdk_client=make_sdk(create))


async def test_generate_normaliza_la_respuesta() -> None:
    create = AsyncMock(return_value=make_completion())
    response = await make_client(create).generate(MESSAGES)

    assert response.content == "Una medida del desorden."
    assert response.provider == "openai"
    assert response.model == "gpt-4o-mini-2024"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.total_tokens == 18


async def test_generate_arma_los_parametros_del_sdk() -> None:
    create = AsyncMock(return_value=make_completion())
    config = GenerationConfig(model="gpt-4.1", temperature=0.3, max_tokens=50)

    await make_client(create).generate(MESSAGES, config)

    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "gpt-4.1"
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_completion_tokens"] == 50
    assert kwargs["messages"] == [
        {"role": "system", "content": "Sé breve."},
        {"role": "user", "content": "¿Qué es la entropía?"},
    ]


async def test_generate_omite_temperatura_si_no_se_define() -> None:
    create = AsyncMock(return_value=make_completion())
    await make_client(create).generate(MESSAGES)

    assert "temperature" not in create.await_args.kwargs
    assert create.await_args.kwargs["model"] == "gpt-4o-mini"


async def test_stream_produce_fragmentos_y_cierra_el_stream() -> None:
    sdk_stream = FakeOpenAIStream(
        [openai_chunk("Hola"), openai_chunk(None), openai_chunk(" mundo")]
    )
    create = AsyncMock(return_value=sdk_stream)

    chunks = [chunk async for chunk in make_client(create).stream(MESSAGES)]

    assert chunks == ["Hola", " mundo"]
    assert create.await_args.kwargs["stream"] is True
    assert sdk_stream.closed


async def test_stream_ignora_chunks_sin_choices() -> None:
    create = AsyncMock(
        return_value=FakeOpenAIStream([SimpleNamespace(choices=[]), openai_chunk("x")])
    )
    chunks = [chunk async for chunk in make_client(create).stream(MESSAGES)]
    assert chunks == ["x"]


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (sdk_status_error(openai.RateLimitError, 429), LLMRateLimitError),
        (sdk_status_error(openai.AuthenticationError, 401), LLMAuthenticationError),
        (sdk_status_error(openai.PermissionDeniedError, 403), LLMAuthenticationError),
        (sdk_status_error(openai.BadRequestError, 400), LLMProviderError),
        (sdk_request_error(openai.APIConnectionError), LLMConnectionError),
        (sdk_request_error(openai.APITimeoutError), LLMTimeoutError),
    ],
)
async def test_generate_traduce_errores_del_sdk(sdk_error: Exception, expected: type) -> None:
    client = make_client(AsyncMock(side_effect=sdk_error))

    with pytest.raises(expected) as exc_info:
        await client.generate(MESSAGES)

    assert exc_info.value.provider == "openai"
    assert exc_info.value.__cause__ is sdk_error


async def test_error_5xx_es_reintentable() -> None:
    client = make_client(AsyncMock(side_effect=sdk_status_error(openai.InternalServerError, 503)))

    with pytest.raises(LLMProviderError) as exc_info:
        await client.generate(MESSAGES)

    assert exc_info.value.status_code == 503
    assert exc_info.value.retryable


async def test_generate_respeta_el_timeout() -> None:
    async def slow_create(**_: object) -> SimpleNamespace:
        return await slow(1, make_completion())

    client = make_client(AsyncMock(side_effect=slow_create))

    with pytest.raises(LLMTimeoutError):
        await client.generate(MESSAGES, GenerationConfig(timeout_s=0.01))


async def test_stream_respeta_el_timeout_entre_fragmentos() -> None:
    create = AsyncMock(return_value=FakeOpenAIStream([openai_chunk("a")], delay=1))

    with pytest.raises(LLMTimeoutError):
        async for _ in make_client(create).stream(MESSAGES, GenerationConfig(timeout_s=0.01)):
            pass


async def test_stream_traduce_errores_a_mitad_de_camino() -> None:
    error = sdk_request_error(openai.APIConnectionError)
    create = AsyncMock(return_value=FakeOpenAIStream([openai_chunk("parcial")], error=error))
    received: list[str] = []

    with pytest.raises(LLMConnectionError):
        async for chunk in make_client(create).stream(MESSAGES):
            received.append(chunk)

    assert received == ["parcial"]


async def test_context_manager_cierra_el_cliente() -> None:
    sdk = make_sdk(AsyncMock())
    async with OpenAIClient(api_key=SecretStr("sk-test"), sdk_client=sdk):
        pass
    sdk.close.assert_awaited_once()
