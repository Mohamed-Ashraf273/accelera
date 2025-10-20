from langchain_google_genai import ChatGoogleGenerativeAI

from mainera.src.models.llm import Llm


class Gemini(Llm):
    def __init__(self, model_name: str, api_key: str, temperature=None):
        super().__init__(model_name, api_key=api_key, temperature=temperature)
        self.model = ChatGoogleGenerativeAI(
            model=self.model_name, api_key=api_key, temperature=self.temperature
        )
