import json
import pickle
from typing import Any
from typing import List

import numpy as np
from schema_validation import InputSchema
from tracking import PredictionTracker


class ModelService:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self._model = None
        self._preprocessors: List[Any] = []
        self._loaded = False
        self.schema = InputSchema()
        self.config = {}
        self.tracker = PredictionTracker()

    @property
    def loaded(self):
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            self.config = cfg
        self.schema = InputSchema(cfg.get("schema"))
        models = cfg.get("models", {})
        self.tracker = PredictionTracker(cfg.get("tracking"))
        model_obj = None
        preprocessors = []
        for _, path in models.items():
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if hasattr(obj, "predict"):
                model_obj = obj
            elif hasattr(obj, "transform"):
                preprocessors.append(obj)

        if model_obj is None:
            raise RuntimeError("No predict-capable model artifact found")

        self._model = model_obj
        self._preprocessors = preprocessors
        self._loaded = True

    def validate_input(self, input_data):
        if not self._loaded:
            self.load()
        return self.schema.validate(input_data)

    def predict(self, input_data, validate=True):
        if not self._loaded:
            self.load()
        rows = self.validate_input(input_data) if validate else input_data
        X = np.array(rows)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        for p in self._preprocessors:
            X = p.transform(X)

        return self._model.predict(X)


service = ModelService()
