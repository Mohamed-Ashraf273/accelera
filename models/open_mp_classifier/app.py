import pickle
from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi import Response
from pydantic import BaseModel

app = FastAPI()

# Load model and encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


class EmbeddingInput(BaseModel):
    embedding: List[float]


class PredictionOutput(BaseModel):
    result: str


@app.post("/predict", response_model=PredictionOutput)
def predict(data: EmbeddingInput):
    embedding = np.array(data.embedding).reshape(1, -1)
    pred = model.predict(embedding)
    pred_class = le.inverse_transform(pred)[0]
    return {"result": pred_class}


@app.get("/")
def read_root():
    return {"message": "OpenMP Loop Classifier API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.head("/health")
def health_head():
    return Response(status_code=200)
