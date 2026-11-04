import pytest
from pydantic import ValidationError

from llm_client import LLMConfigurationError, Settings

ENV_VARS = (
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
    "LLM_MAX_RETRIES",
    "LLM_TIMEOUT_S",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aísla cada test de las variables reales de la máquina."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_from_env_lee_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", " Anthropic ")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LLM_MAX_RETRIES", "4")
    monkeypatch.setenv("LLM_TIMEOUT_S", "12.5")

    settings = Settings.from_env(env_file=None)

    assert settings.provider == "anthropic"
    assert settings.api_key_for("anthropic").get_secret_value() == "sk-ant-test"
    assert settings.max_retries == 4
    assert settings.timeout_s == 12.5


def test_from_env_carga_archivo_dotenv(tmp_path) -> None:
    # `clean_env` registró la variable en monkeypatch, que la restaura al terminar
    # aunque `load_dotenv` la haya escrito en el entorno real.
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-desde-archivo\n", encoding="utf-8")

    settings = Settings.from_env(env_file=env_file)

    assert settings.configured_providers() == ["openai"]


def test_variables_vacias_se_ignoran(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    settings = Settings.from_env(env_file=None)
    assert settings.openai_api_key is None
    assert settings.configured_providers() == []


def test_proveedor_invalido() -> None:
    with pytest.raises(ValidationError):
        Settings(provider="gemini")


def test_falta_api_key_da_error_claro() -> None:
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        Settings().api_key_for("openai")


def test_api_key_no_se_filtra_en_repr() -> None:
    settings = Settings(openai_api_key="sk-super-secreta")
    assert "sk-super-secreta" not in repr(settings)
    assert "sk-super-secreta" not in str(settings)
