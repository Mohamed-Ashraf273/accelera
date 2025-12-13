from langchain_community.llms.ollama import Ollama as OllamaModel

from accelera.src.models.llm import Llm


class Ollama(Llm):
    def __init__(
        self, model_name: str, api_key: str = None, temperature: float = 0.1
    ):
        super().__init__(model_name, api_key, temperature)
        self.model = OllamaModel(
            model=self.model_name, temperature=self.temperature
        )
