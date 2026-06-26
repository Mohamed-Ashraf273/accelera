import io
import time
from contextlib import asynccontextmanager
from typing import Any
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from gui import _render_gui
from modelservice import service
from pydantic import BaseModel
from schema_validation import SchemaValidationError


class PredictPayload(BaseModel):
    input: Optional[Any] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/gui")


@app.get("/gui", response_class=HTMLResponse)
def gui():
    schema = service.schema.describe()
    return HTMLResponse(_render_gui(schema))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": service.loaded,
        "schema": service.schema.describe(),
        "tracking": service.tracker.describe(),
    }


@app.get("/tracking/summary")
def tracking_summary():
    return service.tracker.summary()


@app.post("/predict")
def predict(payload: PredictPayload):
    if payload.input is not None:
        return _predict(payload.input, endpoint="/predict")

    raise HTTPException(status_code=400, detail="No input")


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="file must be CSV")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if df.empty:
        raise HTTPException(status_code=400, detail="file is empty")

    return _predict(df, endpoint="/predict/csv", filename=file.filename)


def _predict(data, endpoint, filename=None):
    started = time.perf_counter()
    rows = []

    try:
        rows = service.validate_input(data)
        preds = service.predict(rows, validate=False)
        predictions = _prediction_values(preds)
        row_count = _row_count(rows)
        _record_prediction(
            endpoint=endpoint,
            status="success",
            rows=row_count,
            latency_ms=_latency_ms(started),
            filename=filename,
            predictions=predictions,
        )
        response = {"rows": row_count, "predictions": predictions}
        if filename:
            response["filename"] = filename
        return response
    except SchemaValidationError as e:
        _record_prediction(
            endpoint=endpoint,
            status="validation_error",
            rows=_row_count(rows),
            latency_ms=_latency_ms(started),
            filename=filename,
            error=str(e),
        )
        raise HTTPException(status_code=422, detail=e.errors)
    except Exception as e:
        _record_prediction(
            endpoint=endpoint,
            status="error",
            rows=_row_count(rows),
            latency_ms=_latency_ms(started),
            filename=filename,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="internal serer error")


def _record_prediction(
    endpoint,
    status,
    rows,
    latency_ms,
    filename=None,
    predictions=None,
    error=None,
):
    event = {
        "endpoint": endpoint,
        "status": status,
        "rows": rows,
        "latency_ms": latency_ms,
    }
    if filename:
        event["filename"] = filename
    if predictions is not None:
        event["predictions"] = predictions
    if error:
        event["error"] = error
    service.tracker.record(event)


def _row_count(rows):
    if rows is None:
        return 0
    if isinstance(rows, pd.DataFrame):
        return len(rows)
    if isinstance(rows, list):
        if rows and all(not isinstance(item, (list, tuple, dict)) for item in rows):
            return 1
        return len(rows)
    try:
        return len(rows)
    except TypeError:
        return 1


def _prediction_values(preds):
    if hasattr(preds, "tolist"):
        return _json_value(preds.tolist())
    if isinstance(preds, list):
        return _json_value(preds)
    if isinstance(preds, tuple):
        return _json_value(list(preds))
    return [_json_value(preds)]


def _json_value(value):
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _latency_ms(started):
    return round((time.perf_counter() - started) * 1000, 2)
