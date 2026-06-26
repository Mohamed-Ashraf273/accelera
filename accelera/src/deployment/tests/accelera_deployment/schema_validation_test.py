import importlib
import sys
from types import SimpleNamespace

# import os
import pytest


class FakeDataFrame:
    def __init__(self, data=None, columns=None):
        self.data = data
        if columns is not None:
            self.columns = list(columns)
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            key = []
            for row in data:
                key.extend(row.keys())
            self.columns = list(dict.fromkeys(key))
        else:
            self.columns = []


class FakeBatchDefinition:
    def get_batch(self, batch_parameters):
        return SimpleNamespace(
            validate=lambda expectation: SimpleNamespace(success=True)
        )


class FakeAsset:
    def add_batch_definition_whole_dataframe(self, _name):
        return FakeBatchDefinition()


class FakeDataSource:
    def add_dataframe_asset(self, name):
        return FakeAsset()


# class FakeSchema:
#     def add_scheam_asset(self, name):
#         return FakeSchema()


class FakeDataSources:
    def add_pandas(self, name):
        return FakeDataSource()


class FakeGXContext:
    data_sources = FakeDataSources()


@pytest.fixture
def schema_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "great_expectations",
        SimpleNamespace(
            get_context=lambda: FakeGXContext(),
            expectations=SimpleNamespace(
                ExpectColumnValuesToNotBeNull=object,
                ExpectColumnValuesToBeBetween=object,
                ExpectColumnValuesToBeInSet=object,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "pandas",
        SimpleNamespace(
            DataFrame=FakeDataFrame,
            to_numeric=lambda values, errors=None: values,
        ),
    )
    sys.modules.pop(
        "accelera.src.deployment.schema_validation",
        None,
    )

    return importlib.import_module("accelera.src.deployment.schema_validation")


# def test_disabled_schema_describes_and_passes_input_through(schema_module):
#     schema = schema_module.InputSchema()
#     data = {"anything": "goes"}
#     assert schema.describe()


def test_enabled_schema_describes_configured__features(schema_module):
    features = [{"name": "age", "type": "integer"}]
    schema = schema_module.InputSchema({"features": features})
    assert schema.describe() == {"enabled": True, "features": features}


def test_validate_missing_columns(schema_module):
    schema = schema_module.InputSchema(
        {"features": [{"name": "age"}, {"name": "income"}]}
    )
    with pytest.raises(schema_module.SchemaValidationError) as exc:
        schema.validate({"age": 30})
    assert exc.value.errors == ["missing columns ['income']"]
    assert str(exc.value) == "missing columns ['income']"


def test_validate_reports(schema_module):
    schema = schema_module.InputSchema({"features": [{"name": "age"}]})
    with pytest.raises(schema_module.SchemaValidationError) as exc:
        schema.validate({"age": 30, "extra": True})
    assert exc.value.errors == ["unexpected columns ['extra']"]
