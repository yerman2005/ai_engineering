import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest
from pydantic import SecretStr

from llm_client import (
    AnthropicClient,
    ChatMessage,
    GenerationConfig,
    LLMConfigurationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
    Role,
)
from tests.fakes import FakeAnthropicStream, sdk_request_error, sdk_status_error

MESSAGES = [
    ChatMessage(role=Role.SYSTEM, content="Sé breve."),
    ChatMessage(role=Role.USER, content="¿Qué es la entropía?"),
]


def make_message() -> SimpleNamespace:
    return SimpleNamespace(
        model="claude-sonnet-5-5",
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Una medida "),
            SimpleNamespace(type="text", text="del desorden."),
        ],
        usage=SimpleNamespace(input_tokens=20, output_tokens=8),
    )


def make_sdk(
    *, create: AsyncMock | None = None, stream: MagicMock | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        messages=SimpleNamespace(create=create or AsyncMock(), stream=stream or MagicMock()),
        close=AsyncMock(),
    )


def make_client(sdk: SimpleNamespace) -> AnthropicClient:
    return AnthropicClient(api_key=SecretStr("sk-ant-test"), sdk_client=sdk)


async def test_generate_toma_solo_los_bloques_de_texto() -> None:
    sdk = make_sdk(create=AsyncMock(return_value=make_message()))

    response = await make_client(sdk).generate(MESSAGES)

    assert response.content == "Una medida del desorden."
    assert response.provider == "anthropic"
    assert response.finish_reason == "end_turn"
    assert response.usage is not None
    assert response.usage.total_tokens == 28


async def test_generate_separa_el_prompt_de_sistema() -> None:
    create = AsyncMock(return_value=make_message())

    await make_client(make_sdk(create=create)).generate(MESSAGES, GenerationConfig(max_tokens=200))

    kwargs = create.await_args.kwargs
    assert kwargs["system"] == "Sé breve."
    assert kwargs["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
    assert kwargs["max_tokens"] == 200
    assert kwargs["model"] == "claude-sonnet-5-5"


async def test_temperatura_no_se_envia_y_se_avisa(caplog: pytest.LogCaptureFixture) -> None:
    create = AsyncMock(return_value=make_message())

    with caplog.at_level(logging.WARNING):
        await make_client(make_sdk(create=create)).generate(
            MESSAGES, GenerationConfig(temperature=1.5)
        )

    assert "temperature" not in create.await_args.kwargs
    assert "temperature" in caplog.text


async def test_solo_mensajes_de_sistema_es_error_de_configuracion() -> None:
    client = make_client(make_sdk())

    with pytest.raises(LLMConfigurationError):
        await client.generate([ChatMessage(role=Role.SYSTEM, content="Solo sistema")])


async def test_lista_de_mensajes_vacia_es_error_de_configuracion() -> None:
    with pytest.raises(LLMConfigurationError):
        await make_client(make_sdk()).generate([])


async def test_stream_produce_fragmentos_y_cierra_el_stream() -> None:
    sdk_stream = FakeAnthropicStream(["Hola", " mundo"])
    stream = MagicMock(return_value=sdk_stream)

    chunks = [chunk async for chunk in make_client(make_sdk(stream=stream)).stream(MESSAGES)]

    assert chunks == ["Hola", " mundo"]
    assert stream.call_args.kwargs["system"] == "Sé breve."
    assert sdk_stream.closed


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (sdk_status_error(anthropic.RateLimitError, 429), LLMRateLimitError),
        (sdk_request_error(anthropic.APIConnectionError), LLMConnectionError),
        (sdk_request_error(anthropic.APITimeoutError), LLMTimeoutError),
    ],
)
async def test_generate_traduce_errores_del_sdk(sdk_error: Exception, expected: type) -> None:
    client = make_client(make_sdk(create=AsyncMock(side_effect=sdk_error)))

    with pytest.raises(expected) as exc_info:
        await client.generate(MESSAGES)

    assert exc_info.value.provider == "anthropic"


async def test_stream_traduce_errores_del_sdk() -> None:
    error = sdk_status_error(anthropic.RateLimitError, 429)
    stream = MagicMock(return_value=FakeAnthropicStream(["parcial"], error=error))

    with pytest.raises(LLMRateLimitError):
        async for _ in make_client(make_sdk(stream=stream)).stream(MESSAGES):
            pass
