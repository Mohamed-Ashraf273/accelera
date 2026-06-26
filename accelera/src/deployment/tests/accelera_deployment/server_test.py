import importlib
import sys

# import json
from types import SimpleNamespace

import pytest


class FakeHTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class FakeFastAPI:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get(self, *_args, **_kwargs):
        return lambda func: func

    def post(self, *_args, **_kwargs):
        return lambda func: func


class FakeHTMLResponse(str):
    pass


class FakeRedirectResponse:
    def __init__(self, url):
        self.url = url


class FakeDataFrame:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.empty = not self.rows

    def __len__(self):
        return len(self.rows)


class FakePredictions:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


# class FakeExpectations:
#     def __init__(self, values):
#         self.values = values
#     def tolist(self):
#         return self.values
class FakeTracker:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)

    def describe(self):
        return {"enabled": True}

    def summary(self):
        return {"total_requests": len(self.events)}


class FakeSchema:
    def describe(self):
        return {"enabled": False, "features": []}


class FakeService:
    def __init__(self):
        self.loaded = True
        self.schema = FakeSchema()
        self.tracker = FakeTracker()
        self.validated_rows = [[1, 2]]
        self.predictions = [3]
        self.validate_error = None
        self.predict_error = None
        self.load_calls = 0

    def load(self):
        self.load_calls += 1

    def validate_input(self, data):
        if self.validate_error:
            raise self.validate_error
        return self.validated_rows

    def predict(self, rows, validate=True):
        if self.predict_error:
            raise self.predict_error
        return FakePredictions(self.predictions)


class FakeSchemaValidationError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(errors))


@pytest.fixture
def server_module(monkeypatch):
    service = FakeService()
    monkeypatch.setitem(
        sys.modules,
        "fastapi",
        SimpleNamespace(
            FastAPI=FakeFastAPI,
            File=lambda *args, **kwargs: None,
            HTTPException=FakeHTTPException,
            UploadFile=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "fastapi.responses",
        SimpleNamespace(
            HTMLResponse=FakeHTMLResponse,
            RedirectResponse=FakeRedirectResponse,
        ),
    )

    monkeypatch.setitem(
        sys.modules,
        "pandas",
        SimpleNamespace(
            DataFrame=FakeDataFrame,
            read_csv=lambda _file: FakeDataFrame([[1, 2]]),
        ),
    )
    monkeypatch.setitem(sys.modules, "pydantic", SimpleNamespace(BaseModel=object))
    monkeypatch.setitem(
        sys.modules,
        "accelera.src.deployment.modelservice",
        SimpleNamespace(service=service),
    )
    monkeypatch.setitem(
        sys.modules,
        "modelservice",
        SimpleNamespace(service=service),
    )
    monkeypatch.setitem(
        sys.modules,
        "accelera.src.deployment.schema_validation",
        SimpleNamespace(SchemaValidationError=FakeSchemaValidationError),
    )
    monkeypatch.setitem(
        sys.modules,
        "schema_validation",
        SimpleNamespace(SchemaValidationError=FakeSchemaValidationError),
    )
    sys.modules.pop("accelera.src.deployment.server", None)
    sys.modules.pop("server", None)

    module = importlib.import_module("accelera.src.deployment.server")
    module.service = service
    sys.modules["server"] = module
    return module


# def test_index_redirects_to_gui(server_module):
#     response = server_module.index
#     assert response.url == "/gui"


def test_health_and_tracking_summary_read_service_state(server_module):
    assert server_module.health() == {
        "status": "ok",
        "model_loaded": True,
        "schema": {"enabled": False, "features": []},
        "tracking": {"enabled": True},
    }
    assert server_module.tracking_summary() == {"total_requests": 0}


def test_predict_rejects_missing_input(server_module):
    with pytest.raises(FakeHTTPException) as exc:
        server_module.predict(SimpleNamespace(input=None))

    assert exc.value.status_code == 400
    assert exc.value.detail == "No input"


def test_predict_success_records_event(server_module):
    response = server_module._predict([[1, 2]], endpoint="/predict")

    assert response == {"rows": 1, "predictions": [3]}
    event = server_module.service.tracker.events[-1]
    assert event["endpoint"] == "/predict"
    assert event["status"] == "success"
    assert event["rows"] == 1
    assert event["predictions"] == [3]
    assert "latency_ms" in event


def test_predict_success_includes_filename(server_module):
    response = server_module._predict(
        FakeDataFrame([[1], [2]]),
        endpoint="/predict/csv",
        filename="input.csv",
    )

    assert response["filename"] == "input.csv"
    assert server_module.service.tracker.events[-1]["filename"] == "input.csv"


def test_predict_validation_error_records_and_raises_422(server_module):
    server_module.service.validate_error = FakeSchemaValidationError(["bad schema"])

    with pytest.raises(FakeHTTPException) as exc:
        server_module._predict([[1, 2]], endpoint="/predict")

    assert exc.value.status_code == 422
    assert exc.value.detail == ["bad schema"]
    event = server_module.service.tracker.events[-1]
    assert event["status"] == "validation_error"
    assert event["error"] == "bad schema"


def test_predict_internal_error_records_and_raises_500(server_module):
    server_module.service.predict_error = RuntimeError("err")
    with pytest.raises(FakeHTTPException) as exc:
        server_module._predict([[1, 2]], endpoint="/predict")
    assert exc.value.status_code == 500
    assert exc.value.detail == "internal serer error"
    event = server_module.service.tracker.events[-1]
    assert event["status"] == "error"
    assert event["error"] == "err"


def test_row_count_handles_supported_inputs(server_module):
    assert server_module._row_count(None) == 0
    assert server_module._row_count([1, 2, 3]) == 1
    assert server_module._row_count([[1], [2]]) == 2
    assert server_module._row_count(FakeDataFrame([[1], [2], [3]])) == 3
    assert server_module._row_count(object()) == 1


def test_render_gui_embeds_schema(server_module):
    html = server_module._render_gui({"enabled": False, "features": []})

    assert "Accelera Deployment" in html
    assert '"enabled": false' in html
