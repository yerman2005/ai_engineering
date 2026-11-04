"""Unified Async LLM Client: una interfaz común y asíncrona para OpenAI y Anthropic."""

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .config import SUPPORTED_PROVIDERS, ProviderName, Settings
from .exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .manager import AsyncLLMManager
from .openai_client import OpenAIClient
from .schemas import (
    ChatMessage,
    ErrorInfo,
    GenerationConfig,
    LLMResult,
    ModelResponse,
    Role,
    TokenUsage,
)

__all__ = [
    "SUPPORTED_PROVIDERS",
    "AnthropicClient",
    "AsyncLLMManager",
    "BaseLLMClient",
    "ChatMessage",
    "ErrorInfo",
    "GenerationConfig",
    "LLMAuthenticationError",
    "LLMConfigurationError",
    "LLMConnectionError",
    "LLMError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResult",
    "LLMTimeoutError",
    "ModelResponse",
    "OpenAIClient",
    "ProviderName",
    "Role",
    "Settings",
    "TokenUsage",
]
