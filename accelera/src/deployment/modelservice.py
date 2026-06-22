import os
import json
import pickle
from typing import Any
from typing import List
import torch
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
        config_dir = os.path.dirname(os.path.abspath(self.config_path))
        for _, path in models.items():
            abs_path = os.path.join(config_dir, path) if not os.path.isabs(path) else path
            if abs_path.endswith(".pth") or abs_path.endswith(".pt"):
                obj = torch.load(abs_path, map_location=torch.device('cpu'), weights_only=False)
                model_obj = obj
            else:
                with open(abs_path, "rb") as f:
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

        if hasattr(self._model, "predict"):
            return self._model.predict(X)

        import torch
        tensor_X = torch.tensor(X, dtype=torch.float32)
        if hasattr(self._model, "eval"):
            self._model.eval()
        with torch.no_grad():
            outputs = self._model(tensor_X)

        if isinstance(outputs, torch.Tensor):
            preds = outputs.cpu().numpy()
        else:
            preds = np.array(outputs)

        if preds.ndim > 1 and preds.shape[1] > 1:
            return preds.argmax(axis=1)
        if preds.ndim > 1 and preds.shape[1] == 1:
            return preds.flatten()
        return preds


service = ModelService()
