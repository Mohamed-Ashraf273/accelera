import importlib
import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest


class DummySchema:
    instances = []

    def __init__(self, config=None):
        self.config = config
        self.validated = []
        DummySchema.instances.append(self)

    def validate(self, input_data):
        self.validated.append(input_data)
        return input_data

    def describe(self):
        return {"enabled": bool(self.config)}


class DummyTracker:
    instances = []

    def __init__(self, config=None):
        self.config = config
        DummyTracker.instances.append(self)

    def describe(self):
        return {"enabled": bool(self.config)}


# class DummyTrans:
#     def transform(self, X):
#         return X + 1


class DummyPreprocessor:
    def transform(self, X):
        return X + 1


class DummyModel:
    def predict(self, X):
        return np.asarray(X).sum(axis=1)


@pytest.fixture
def service_module(monkeypatch):
    DummySchema.instances.clear()
    DummyTracker.instances.clear()

    monkeypatch.setitem(
        sys.modules,
        "accelera.src.deployment.schema_validation",
        SimpleNamespace(InputSchema=DummySchema),
    )
    monkeypatch.setitem(
        sys.modules,
        "schema_validation",
        SimpleNamespace(InputSchema=DummySchema),
    )
    monkeypatch.setitem(
        sys.modules,
        "accelera.src.deployment.tracking",
        SimpleNamespace(PredictionTracker=DummyTracker),
    )
    monkeypatch.setitem(
        sys.modules,
        "tracking",
        SimpleNamespace(PredictionTracker=DummyTracker),
    )
    sys.modules.pop("accelera.src.deployment.modelservice", None)
    sys.modules.pop("modelservice", None)
    module = importlib.import_module("accelera.src.deployment.modelservice")
    sys.modules["modelservice"] = module
    return module


def write_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def test_load_reads_config_and_artifacts(service_module, tmp_path):
    preprocessor_path = tmp_path / "preprocessor.pkl"
    model_path = tmp_path / "model.pkl"
    config_path = tmp_path / "config.json"
    write_pickle(preprocessor_path, DummyPreprocessor())
    write_pickle(model_path, DummyModel())
    config_path.write_text(
        json.dumps(
            {
                "schema": {"features": [{"name": "x"}]},
                "tracking": {"enabled": True},
                "models": {
                    "preprocessor": str(preprocessor_path),
                    "model": str(model_path),
                },
            }
        ),
        encoding="utf-8",
    )
    service = service_module.ModelService(config_path=str(config_path))
    service.load()
    assert service.loaded is True
    assert service.config["models"]["model"] == str(model_path)
    assert len(service._preprocessors) == 1
    assert isinstance(service._model, DummyModel)
    assert DummySchema.instances[-1].config == {"features": [{"name": "x"}]}
    assert DummyTracker.instances[-1].config == {"enabled": True}


def test_load_is_idempotent(service_module, tmp_path, monkeypatch):
    model_path = tmp_path / "model.pkl"
    config_path = tmp_path / "config.json"
    # expr_path = tmp_path / "expr.json"
    write_pickle(model_path, DummyModel())
    config_path.write_text(
        json.dumps({"models": {"model": str(model_path)}}),
        encoding="utf-8",
    )
    service = service_module.ModelService(config_path=str(config_path))
    service.load()

    def fail(*_args, **_kwargs):
        raise AssertionError("load should not reopen files once loaded")

    monkeypatch.setattr(service_module, "open", fail, raising=False)
    service.load()
    assert service.loaded is True


def test_load_requires_predict_capable_artifact(service_module, tmp_path):
    preprocessor_path = tmp_path / "preprocessor.pkl"
    config_path = tmp_path / "config.json"
    write_pickle(preprocessor_path, DummyPreprocessor())
    config_path.write_text(
        json.dumps({"models": {"preprocessor": str(preprocessor_path)}}),
        encoding="utf-8",
    )
    service = service_module.ModelService(config_path=str(config_path))

    with pytest.raises(RuntimeError, match="no model with predict attr found"):
        service.load()


def test_validate_input_loads_service_before_validating(service_module, tmp_path):
    model_path = tmp_path / "model.pkl"
    config_path = tmp_path / "config.json"
    write_pickle(model_path, DummyModel())
    config_path.write_text(
        json.dumps({"models": {"model": str(model_path)}}),
        encoding="utf-8",
    )
    service = service_module.ModelService(config_path=str(config_path))

    rows = service.validate_input([[1, 2]])

    assert rows == [[1, 2]]
    assert DummySchema.instances[-1].validated == [[[1, 2]]]


def test_predict_validates_preprocesses_and_returns_predictions(
    service_module,
    tmp_path,
):
    preprocessor_path = tmp_path / "preprocessor.pkl"
    model_path = tmp_path / "model.pkl"
    config_path = tmp_path / "config.json"
    write_pickle(preprocessor_path, DummyPreprocessor())
    write_pickle(model_path, DummyModel())
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "preprocessor": str(preprocessor_path),
                    "model": str(model_path),
                }
            }
        ),
        encoding="utf-8",
    )
    service = service_module.ModelService(config_path=str(config_path))
    service.load()

    predictions = service.predict([1, 2])

    assert predictions.tolist() == [5]
    assert DummySchema.instances[-1].validated == [[1, 2]]


# def (service_module):
#     service = service_module.ModelService()
#     service._loaded = True
#     service.schema = DummySchema()
#     service._model = DummyModel()
#     service._preprocessors = []

#     predictions = service.predict([[3, 4]], validate=False)

#     assert predictions.tolist() == [7]
#     assert service.schema.validated == []
