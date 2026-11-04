import pytest
from pydantic import SecretStr, ValidationError

from llm_client import (
    AnthropicClient,
    AsyncLLMManager,
    ChatMessage,
    GenerationConfig,
    LLMConfigurationError,
    LLMRateLimitError,
    OpenAIClient,
    Role,
    Settings,
)
from tests.fakes import FakeLLMClient

SETTINGS = Settings(
    provider="openai",
    openai_api_key=SecretStr("sk-test"),
    anthropic_api_key=SecretStr("sk-ant-test"),
    anthropic_model="claude-haiku-5-5",
)


class TestFactory:
    @pytest.mark.parametrize(
        ("provider", "expected_cls", "expected_model"),
        [
            ("openai", OpenAIClient, "gpt-4o-mini"),
            ("anthropic", AnthropicClient, "claude-haiku-5-5"),
        ],
    )
    async def test_crea_el_cliente_segun_la_configuracion(
        self, provider: str, expected_cls: type, expected_model: str
    ) -> None:
        async with AsyncLLMManager.from_settings(SETTINGS, provider) as manager:
            assert isinstance(manager._client, expected_cls)
            assert manager._client.default_model == expected_model
            assert manager.provider == provider

    async def test_usa_el_proveedor_por_defecto_de_settings(self) -> None:
        async with AsyncLLMManager.from_settings(SETTINGS) as manager:
            assert manager.provider == "openai"

    def test_falta_api_key(self) -> None:
        with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
            AsyncLLMManager.from_settings(Settings(), "anthropic")

    def test_proveedor_desconocido(self) -> None:
        with pytest.raises(LLMConfigurationError):
            AsyncLLMManager.from_settings(SETTINGS, "gemini")  # type: ignore[arg-type]


async def test_generate_con_prompt_simple_y_sistema() -> None:
    client = FakeLLMClient(reply="ok")
    manager = AsyncLLMManager(client)

    result = await manager.generate("Hola", system="Sé breve.")

    assert result.ok
    assert result.response is not None
    assert result.response.content == "ok:Hola"
    assert client.calls[0] == [
        ChatMessage(role=Role.SYSTEM, content="Sé breve."),
        ChatMessage(role=Role.USER, content="Hola"),
    ]


async def test_generate_acepta_lista_de_mensajes() -> None:
    client = FakeLLMClient()
    history = [
        ChatMessage(role=Role.USER, content="Hola"),
        ChatMessage(role=Role.ASSISTANT, content="¡Hola!"),
        ChatMessage(role=Role.USER, content="¿Qué es un LLM?"),
    ]

    await AsyncLLMManager(client).generate(history)

    assert client.calls[0] == history


async def test_generate_devuelve_error_controlado_en_lugar_de_lanzar() -> None:
    client = FakeLLMClient(error=LLMRateLimitError("Demasiadas solicitudes", provider="fake"))

    result = await AsyncLLMManager(client).generate("Hola")

    assert not result.ok
    assert result.error is not None
    assert result.error.code == "rate_limit_error"
    assert result.error.retryable


async def test_generate_valida_el_prompt_vacio() -> None:
    with pytest.raises(ValidationError):
        await AsyncLLMManager(FakeLLMClient()).generate("")


async def test_stream_reexpone_los_fragmentos() -> None:
    manager = AsyncLLMManager(FakeLLMClient(chunks=["a", "b", "c"]))
    assert [chunk async for chunk in manager.stream("Hola")] == ["a", "b", "c"]


async def test_stream_propaga_llm_error() -> None:
    error = LLMRateLimitError("cortado", provider="fake")
    manager = AsyncLLMManager(FakeLLMClient(chunks=["a"], error=error))

    with pytest.raises(LLMRateLimitError):
        async for _ in manager.stream("Hola"):
            pass


async def test_generate_many_limita_la_concurrencia_y_respeta_el_orden() -> None:
    client = FakeLLMClient(reply="r", delay=0.01)
    prompts = [f"p{i}" for i in range(10)]

    results = await AsyncLLMManager(client).generate_many(prompts, max_concurrency=3)

    assert [r.response.content for r in results if r.response] == [f"r:{p}" for p in prompts]
    assert client.max_active == 3


async def test_generate_many_rechaza_concurrencia_invalida() -> None:
    with pytest.raises(ValueError):
        await AsyncLLMManager(FakeLLMClient()).generate_many(["a"], max_concurrency=0)


async def test_config_por_defecto_y_override() -> None:
    client = FakeLLMClient()
    manager = AsyncLLMManager(client, default_config=GenerationConfig(model="modelo-a"))

    first = await manager.generate("x")
    second = await manager.generate("x", config=GenerationConfig(model="modelo-b"))

    assert first.response is not None and first.response.model == "modelo-a"
    assert second.response is not None and second.response.model == "modelo-b"


async def test_context_manager_cierra_el_cliente() -> None:
    client = FakeLLMClient()
    async with AsyncLLMManager(client):
        pass
    assert client.closed
