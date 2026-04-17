import io
from contextlib import asynccontextmanager
from typing import Any
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from modelservice import service  # the models
from pydantic import BaseModel


class PredictPayload(BaseModel):
    input: Optional[Any] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
def predict(payload: PredictPayload):
    data = None
    if payload.input is not None:
        data = payload.input
    else:
        raise HTTPException(status_code=400, detail="No input provided")

    try:
        preds = service.predict(data)
        return {"predictions": preds.tolist()}
    except Exception:
        raise HTTPException(status_code=500, detail="internal server error")


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="file must be a CSV")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    try:
        preds = service.predict(df.values.tolist())
        return {
            "filename": file.filename,
            "rows": len(df),
            "predictions": preds.tolist(),
        }
    except Exception:
        raise HTTPException(status_code=500, detail="internal server error")
