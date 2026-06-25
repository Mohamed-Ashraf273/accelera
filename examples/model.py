from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from accelera.src.config import config as accelera_config

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


schema = {
    "features": [
        {
            "name": "alcohol",
            "type": "number",
            "required": False,
            "min": 1,
            "max": 15,
        },
        {
            "name": "malic_acid",
            "type": "number",
            "required": False,
            "min": 2,
            "max": 5,
        },
        {"name": "ash", "type": "number", "required": False, "min": 1, "max": 3},
        {
            "name": "alcalinity_of_ash",
            "type": "number",
            "required": False,
            "min": 10,
            "max": 30,
        },
        {
            "name": "magnesium",
            "type": "number",
            "required": False,
            "min": 20,
            "max": 160,
        },
        {
            "name": "total_phenols",
            "type": "number",
            "required": False,
            "min": 1,
            "max": 4,
        },
        {
            "name": "flavanoids",
            "type": "number",
            "required": False,
            "min": 0,
            "max": 6,
        },
        {
            "name": "color_intensity",
            "type": "number",
            "required": False,
            "min": 1,
            "max": 13,
        },
        {"name": "hue", "type": "number", "required": False, "min": 1, "max": 2},
        {
            "name": "proline",
            "type": "number",
            "required": False,
            "min": 278,
            "max": 1680,
        },
        {
            "name": "color_group",
            "type": "string",
            "required": False,
            "allowed_values": ["dark", "light"],
        },
        {
            "name": "proline_class",
            "type": "string",
            "required": False,
            "allowed_values": ["high", "low", "medium"],
        },
    ]
}


def train_model(data=None):
    data = data or {}
    output_dir_str = data.get("output_dir") or str(accelera_config.deployment_root)
    output_dir = Path(output_dir_str).resolve()
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data = load_wine(as_frame=True)
    df = data.frame.copy()
    target_column = data.get("target_column", "target")
    df[target_column] = data.target.map(
        {0: "cultivar_1", 1: "cultivar_2", 2: "cultivar_3"}
    )

    features = [
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "color_intensity",
        "hue",
        "proline",
        target_column,
    ]
    df = df[features]

    df["color_group"] = pd.cut(
        df["color_intensity"], bins=2, labels=["light", "dark"]
    ).astype(str)
    df["proline_class"] = pd.cut(
        df["proline"], bins=3, labels=["low", "medium", "high"]
    ).astype(str)

    X = df.drop(columns=[target_column])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_column])

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    numeric_indices = [X.columns.get_loc(col) for col in numeric_features]
    categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_indices,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical_indices,
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "feature_selector",
                SelectPercentile(score_func=f_classif, percentile=80),
            ),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )
    pipeline.fit(X, y)

    fitted_preprocessor = pipeline.named_steps["preprocess"]
    feature_selector = pipeline.named_steps["feature_selector"]
    model = pipeline.named_steps["model"]

    with open(models_dir / "preprocessing_pipeline.pkl", "wb") as f:
        pickle.dump(fitted_preprocessor, f)
    with open(models_dir / "feature_selector.pkl", "wb") as f:
        pickle.dump(feature_selector, f)
    with open(models_dir / "final_model.pkl", "wb") as f:
        pickle.dump(model, f)

    deploy_config = {
        "models": {
            "preprocessing_pipeline": "models/preprocessing_pipeline.pkl",
            "feature_selector": "models/feature_selector.pkl",
            "final_model": "models/final_model.pkl",
        },
        "schema": schema,
        "tracking": {
            "enabled": True,
            "path": "prediction_logs/predictions.jsonl",
        },
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(deploy_config, f, indent=2)

    print("finished training")
    return {
        "model": "random_forest",
        "deployment_config": deploy_config,
    }


def load_final_pipeline(output_dir=None):
    from sklearn.pipeline import Pipeline

    output_dir_str = output_dir or str(accelera_config.deployment_root)
    output_dir = Path(output_dir_str).resolve()
    models_dir = output_dir / "models"

    with open(models_dir / "preprocessing_pipeline.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open(models_dir / "feature_selector.pkl", "rb") as f:
        selector = pickle.load(f)
    with open(models_dir / "final_model.pkl", "rb") as f:
        model = pickle.load(f)

    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("feature_selector", selector),
            ("model", model),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    train_model({"output_dir": args.output_dir})
