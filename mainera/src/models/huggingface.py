from langchain_huggingface import HuggingFaceEndpoint

from mainera.src.models.llm import Llm


class HuggingFaceModel(Llm):
    def __init__(
        self, model_name: str, api_key: str, provider, temperature: float = 0.1
    ):
        super().__init__(model_name, api_key=api_key, temperature=temperature)

        self.model = HuggingFaceEndpoint(
            repo_id=self.model_name,
            max_new_tokens=1000,
            temperature=self.temperature,
            top_k=10,
            top_p=0.95,
            provider=provider,
            huggingfacehub_api_token=self.api_key,
        )
