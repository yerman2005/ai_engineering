import pytest
from pydantic import ValidationError

from llm_client import ChatMessage, GenerationConfig, LLMResult, ModelResponse, Role, TokenUsage


class TestChatMessage:
    def test_acepta_rol_como_texto(self) -> None:
        message = ChatMessage(role="user", content="Hola")
        assert message.role is Role.USER

    def test_rechaza_contenido_vacio(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(role=Role.USER, content="")

    def test_rechaza_rol_desconocido(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(role="tool", content="Hola")

    def test_es_inmutable(self) -> None:
        message = ChatMessage(role=Role.USER, content="Hola")
        with pytest.raises(ValidationError):
            message.content = "otro"


class TestGenerationConfig:
    def test_valores_por_defecto(self) -> None:
        config = GenerationConfig()
        assert config.temperature is None
        assert config.max_tokens == 1024
        assert config.model is None

    @pytest.mark.parametrize("temperature", [0.0, 1.0, 2.0])
    def test_temperatura_valida(self, temperature: float) -> None:
        assert GenerationConfig(temperature=temperature).temperature == temperature

    @pytest.mark.parametrize("temperature", [-0.1, 2.1])
    def test_temperatura_fuera_de_rango(self, temperature: float) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(temperature=temperature)

    @pytest.mark.parametrize("max_tokens", [0, -5, 128_001])
    def test_max_tokens_fuera_de_rango(self, max_tokens: int) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(max_tokens=max_tokens)

    def test_timeout_debe_ser_positivo(self) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(timeout_s=0)

    def test_rechaza_campos_desconocidos(self) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(top_k=5)


def test_token_usage_total() -> None:
    assert TokenUsage(input_tokens=10, output_tokens=5).total_tokens == 15


def test_llm_result_ok() -> None:
    response = ModelResponse(content="x", provider="openai", model="m")
    assert LLMResult(response=response).ok
    assert not LLMResult().ok
