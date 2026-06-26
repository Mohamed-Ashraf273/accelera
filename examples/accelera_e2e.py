import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from accelera.src.accelera_pipe.core.pipeline import Pipeline
from accelera.src.e2e.tabular.e2e import E2E


def get_random_row(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("CSV file is empty")

    return df.sample(n=1)


config = {"target_col": "price"}

accpipe = Pipeline()
accpipe.preprocess(
    "p1",
    OneHotEncoder(),
    columns=[
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
        "furnishingstatus",
    ],
).model("lr", LinearRegression())

e2e = E2E()
predictions, executed_graph = e2e(
    graph=accpipe,
    content="https://drive.google.com/uc?id=1VMtLcWDcigwkimpf-eWVMZ7zJMUf7wxs",
    config=config,
)

with open("pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

print(
    pipeline.predict(
        get_random_row("/home/mohamed-ashraf/Desktop/projects/accelera/Housing.csv")
    )
)
