class Llm:
    def __init__(
        self, model_name: str, api_key: str = None, temperature: float = 0.1
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.model = None

    def __call__(self, prompt):
        if self.model is None:
            raise ValueError("Model is not initialized.")
        return self.model.invoke(prompt)

    def llm(self):
        if self.model is None:
            raise ValueError("Model is not initialized.")
        return self.model
