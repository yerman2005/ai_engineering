import asyncio
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

load_dotenv()

# 1. Definimos el contrato común (abstracción)
class BaseLLMClient(ABC):
    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        """Envía un prompt al modelo y retorna la respuesta generada."""
        pass

# 2. Implementación concreta para un proveedor real
class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def generate_response(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-5-5"):
        self.api_key = api_key
        self.model = model
        self.client = AsyncAnthropic(api_key=self.api_key)

    async def generate_response(self, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response.content[0].text

# 3. Consumo del cliente sin acoplarse a la implementación específica
async def main():
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    prompts = [
        "¿Qué es un LLM?",
        "Diferencia entre sincrónico y asincrónico",
    ]

    if openai_key:
        openai_client: BaseLLMClient = OpenAIClient(api_key=openai_key)
        print("--- OpenAI ---")
        respuestas = await asyncio.gather(*(openai_client.generate_response(p) for p in prompts))
        for r in respuestas:
            print(r)

    if anthropic_key:
        anthropic_client: BaseLLMClient = AnthropicClient(api_key=anthropic_key)
        print("\n--- Anthropic ---")
        respuestas = await asyncio.gather(*(anthropic_client.generate_response(p) for p in prompts))
        for r in respuestas:
            print(r)

    if not openai_key and not anthropic_key:
        print("No hay claves API configuradas. Define OPENAI_API_KEY o ANTHROPIC_API_KEY.")

if __name__ == "__main__":
    asyncio.run(main())