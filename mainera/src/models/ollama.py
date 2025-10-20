from langchain_community.llms.ollama import Ollama as OllamaModel

from mainera.src.models.llm import Llm


class Ollama(Llm):
    def __init__(self, model_name: str, api_key: str = None, temprature=None):
        super().__init__(model_name, api_key, temprature)
        self.model = OllamaModel(
            model=self.model_name, temperature=self.temprature
        )
