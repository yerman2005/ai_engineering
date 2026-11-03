import asyncio
from abc import ABC, abstractmethod
# 1. Definimos el contrato común (abstracción)
class BaseLLMClient(ABC):
  @abstractmethod
  async def generate_response(self, prompt: str) -> str:
    """Envía un prompt al modelo y retorna la respuesta generada."""
    pass
# 2. Implementación concreta para un proveedor (ejemplo con mock/llamada asíncrona)
class OpenAIClient(BaseLLMClient):
  def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
    self.api_key = api_key
    self.model = model
  async def generate_response(self, prompt: str) -> str:
    # Simulamos la latencia de la llamada a la API
    await asyncio.sleep(1)
    return (
        f"[{self.model}] Respuesta procesada asíncronamente para: '{prompt}'"
    )
# 3. Consumo del cliente sin acoplarse a la implementación específica
async def main():
  client: BaseLLMClient = OpenAIClient(api_key="tu-api-key")
  # Ejecutamos llamadas concurrentes con asyncio.gather
  prompts = [
      "¿Qué es un LLM?",
      "Diferencia entre sincrónico y asincrónico",
  ]
  tareas = [client.generate_response(p) for p in prompts]
  respuestas = await asyncio.gather(*tareas)
  for r in respuestas:
    print(r)
if __name__ == "__main__":
  asyncio.run(main())