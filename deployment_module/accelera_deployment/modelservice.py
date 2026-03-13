import json
import pickle
from typing import Any
from typing import List

import numpy as np


class ModelService:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self._model = None
        self._preprocessors: List[Any] = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        with open(self.config_path, "r") as f:
            cfg = json.load(f)
        models = cfg.get("models", {})
        model_obj = None
        preprocessors = []
        for _, path in models.items():
            obj = pickle.load(open(path, "rb"))
            if hasattr(obj, "predict"):
                model_obj = obj
            elif hasattr(obj, "transform"):
                preprocessors.append(obj)
        self._model = model_obj
        self._preprocessors = preprocessors
        self._loaded = True

    def predict(self, input_data):
        if not self._loaded:
            self.load()
        X = np.array(input_data)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        for p in self._preprocessors:
            X = p.transform(X)
        return self._model.predict(X)


service = ModelService()
