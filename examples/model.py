"""Case-study training pipeline for the deployment module.

The script builds a small but realistic classification pipeline and writes the
artifacts expected by the deployment tooling.  It can train from a local CSV,
from a CSV URL, or from a built-in sklearn dataset with a couple of extra
categorical columns added for preprocessing demos.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from accelera.src.config import config as accelera_config

CONFIG = {
    "dataset_path": None,
    "dataset_url": None,
    "target_column": "target",
    "output_dir": None,
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "feature_selection_percentile": 80,
}


def require_ml_dependencies():
    """Import ML dependencies lazily so missing packages fail with a useful error."""
    try:
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.datasets import load_wine
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectPercentile
        from sklearn.feature_selection import f_classif
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Missing ML dependency. Install the project requirements before "
            "running the deployment training pipeline."
        ) from exc

    return {
        "np": np,
        "pd": pd,
        "ColumnTransformer": ColumnTransformer,
        "load_wine": load_wine,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "SelectPercentile": SelectPercentile,
        "f_classif": f_classif,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline,
        "LabelEncoder": LabelEncoder,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


def save_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


def load_dataset(settings, deps):
    """Load a CSV dataset, or create the built-in case-study dataset."""
    pd = deps["pd"]
    np = deps["np"]
    load_wine = deps["load_wine"]

    dataset_path = settings.get("dataset_path")
    dataset_url = settings.get("dataset_url")
    target_column = settings["target_column"]

    if dataset_path:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return pd.read_csv(path), f"csv:{path}"

    if dataset_url:
        try:
            return pd.read_csv(dataset_url), f"url:{dataset_url}"
        except Exception as exc:
            raise RuntimeError(f"Could not load dataset URL: {dataset_url}") from exc

    data = load_wine(as_frame=True)
    df = data.frame.copy()
    df[target_column] = data.target

    # Map numeric targets to meaningful names
    df[target_column] = df[target_column].map(
        {0: "cultivar_1", 1: "cultivar_2", 2: "cultivar_3"}
    )

    # Keep 10 meaningful features + target_column
    keep_features = [
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
    df = df[keep_features]

    # Add 2 meaningful derived categorical columns for case-study encoding
    df["color_group"] = pd.cut(
        df["color_intensity"],
        bins=2,
        labels=["light", "dark"],
    ).astype(str)

    df["proline_class"] = pd.cut(
        df["proline"],
        bins=3,
        labels=["low", "medium", "high"],
    ).astype(str)

    # Introduce some missing values to test preprocessing pipelines
    rng = np.random.default_rng(settings["random_state"])
    for column in ["alcohol", "malic_acid", "color_group"]:
        missing_rows = rng.choice(
            df.index, size=max(1, len(df) // 25), replace=False
        )
        df.loc[missing_rows, column] = np.nan

    return df, "sklearn:wine_10_features_with_categorical"


def validate_dataset(df, target_column):
    errors = []
    warnings = []

    if target_column not in df.columns:
        errors.append(f"target column {target_column!r} is missing")
    if df.empty:
        errors.append("dataset is empty")
    if df.columns.duplicated().any():
        errors.append("dataset has duplicate column names")
    if target_column in df.columns and df[target_column].isna().any():
        errors.append("target column contains missing values")
    if target_column in df.columns and df[target_column].nunique() < 2:
        errors.append("target column must contain at least two classes")

    missing_counts = df.isna().sum().to_dict()
    if any(count > 0 for count in missing_counts.values()):
        warnings.append("dataset contains missing values; imputers will handle them")

    report = {
        "errors": errors,
        "warnings": warnings,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_counts": {str(k): int(v) for k, v in missing_counts.items()},
    }
    if errors:
        raise ValueError("; ".join(errors))
    return report


def dataset_metadata(df, target_column, source):
    feature_columns = [col for col in df.columns if col != target_column]
    return {
        "source": source,
        "created_at": datetime.now().isoformat(),
        "target_column": target_column,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "feature_columns": feature_columns,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "target_distribution": {
            str(k): int(v) for k, v in df[target_column].value_counts().items()
        },
    }


def split_features_target(df, target_column, deps):
    LabelEncoder = deps["LabelEncoder"]

    X = df.drop(columns=[target_column])
    y_raw = df[target_column]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    label_mapping = {
        int(index): str(label)
        for index, label in enumerate(label_encoder.classes_.tolist())
    }
    return X, y, label_encoder, label_mapping


def make_preprocessor(X, deps):
    ColumnTransformer = deps["ColumnTransformer"]
    OneHotEncoder = deps["OneHotEncoder"]
    Pipeline = deps["Pipeline"]
    SimpleImputer = deps["SimpleImputer"]
    StandardScaler = deps["StandardScaler"]

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    numeric_indices = [X.columns.get_loc(col) for col in numeric_features]
    categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_indices),
            ("cat", categorical_pipeline, categorical_indices),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def model_candidates(settings, deps):
    LogisticRegression = deps["LogisticRegression"]
    RandomForestClassifier = deps["RandomForestClassifier"]
    HistGradientBoostingClassifier = deps["HistGradientBoostingClassifier"]
    random_state = settings["random_state"]

    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=random_state,
        ),
    }


def build_pipeline(preprocessor, model, settings, deps):
    Pipeline = deps["Pipeline"]
    SelectPercentile = deps["SelectPercentile"]
    f_classif = deps["f_classif"]

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "feature_selector",
                SelectPercentile(
                    score_func=f_classif,
                    percentile=settings["feature_selection_percentile"],
                ),
            ),
            ("model", model),
        ]
    )


def evaluate_pipeline(pipeline, X_test, y_test, label_mapping, deps):
    accuracy_score = deps["accuracy_score"]
    classification_report = deps["classification_report"]
    confusion_matrix = deps["confusion_matrix"]
    f1_score = deps["f1_score"]

    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=[label_mapping[i] for i in sorted(label_mapping)],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def feature_names_after_preprocessing(preprocessor):
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []


def selected_feature_names(pipeline, all_feature_names):
    selector = pipeline.named_steps["feature_selector"]
    try:
        mask = selector.get_support()
        return [
            name for name, keep in zip(all_feature_names, mask, strict=False) if keep
        ]
    except Exception:
        return []


def schema_from_features(X):
    features = []
    for column in X.columns:
        if str(X[column].dtype) in {"object", "category"}:
            feature = {
                "name": column,
                "type": "string",
                "required": False,
                "allowed_values": sorted(
                    str(value) for value in X[column].dropna().unique().tolist()
                ),
            }
        else:
            feature = {
                "name": column,
                "type": "number",
                "required": False,
            }
            if X[column].notna().any():
                feature["min"] = float(X[column].min())
                feature["max"] = float(X[column].max())
        features.append(feature)
    return {"features": features}


def relative_to_output(path, output_dir):
    return str(Path(path).resolve().relative_to(Path(output_dir).resolve()))


def write_deployment_config(output_dir, artifact_paths, schema):
    deploy_config = {
        "models": {
            "preprocessing_pipeline": relative_to_output(
                artifact_paths["preprocessing_pipeline"], output_dir
            ),
            "feature_selector": relative_to_output(
                artifact_paths["feature_selector"], output_dir
            ),
            "final_model": relative_to_output(
                artifact_paths["final_model"], output_dir
            ),
        },
        "artifacts": {
            name: relative_to_output(path, output_dir)
            for name, path in artifact_paths.items()
        },
        "schema": schema,
        "tracking": {
            "enabled": True,
            "path": "prediction_logs/predictions.jsonl",
        },
    }
    save_json(deploy_config, Path(output_dir) / "config.json")
    return deploy_config


def train_model(settings=None):
    deps = require_ml_dependencies()
    np = deps["np"]
    train_test_split = deps["train_test_split"]

    settings = {**CONFIG, **(settings or {})}
    output_dir_str = settings.get("output_dir")
    if output_dir_str is None:
        output_dir_str = str(accelera_config.deployment_root)
    output_dir = Path(output_dir_str).resolve()
    models_dir = output_dir / "models"
    artifacts_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)

    print("Loading dataset...")
    df, source = load_dataset(settings, deps)
    validation_report = validate_dataset(df, settings["target_column"])
    metadata = dataset_metadata(df, settings["target_column"], source)
    X, y, label_encoder, label_mapping = split_features_target(
        df,
        settings["target_column"],
        deps,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings["test_size"],
        random_state=settings["random_state"],
        stratify=y,
    )

    preprocessor, numeric_features, categorical_features = make_preprocessor(X, deps)
    candidates = model_candidates(settings, deps)
    comparison = []
    pipelines = {}

    print(f"Training {len(candidates)} model(s)...")
    for name, model in candidates.items():
        candidate_preprocessor, _, _ = make_preprocessor(X, deps)
        pipeline = build_pipeline(candidate_preprocessor, model, settings, deps)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_pipeline(pipeline, X_test, y_test, label_mapping, deps)
        pipelines[name] = pipeline
        comparison.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"],
                "metrics": metrics,
            }
        )
        save_pickle(pipeline, models_dir / f"pipeline_{name}.pkl")
        save_pickle(pipeline.named_steps["model"], models_dir / f"model_{name}.pkl")
        print(f"  {name}: accuracy={metrics['accuracy']:.4f}")

    comparison = sorted(
        comparison,
        key=lambda item: (item["f1_weighted"], item["accuracy"]),
        reverse=True,
    )
    best_name = comparison[0]["model"]
    final_pipeline = pipelines[best_name]
    final_metrics = comparison[0]["metrics"]

    # These fitted pieces are saved separately so they can be reused or inspected.
    fitted_preprocessor = final_pipeline.named_steps["preprocess"]
    feature_selector = final_pipeline.named_steps["feature_selector"]
    final_model = final_pipeline.named_steps["model"]
    all_feature_names = feature_names_after_preprocessing(fitted_preprocessor)
    chosen_features = selected_feature_names(final_pipeline, all_feature_names)

    artifact_paths = {
        "raw_dataset_metadata": artifacts_dir / "raw_dataset_metadata.pkl",
        "validation_report": artifacts_dir / "validation_report.pkl",
        "train_test_split": artifacts_dir / "train_test_split.pkl",
        "label_encoder": artifacts_dir / "label_encoder.pkl",
        "label_mapping": artifacts_dir / "label_mapping.pkl",
        "preprocessing_pipeline": models_dir / "preprocessing_pipeline.pkl",
        "numeric_imputer": models_dir / "numeric_imputer.pkl",
        "categorical_imputer": models_dir / "categorical_imputer.pkl",
        "encoder": models_dir / "encoder.pkl",
        "scaler": models_dir / "scaler.pkl",
        "feature_selector": models_dir / "feature_selector.pkl",
        "trained_models": models_dir / "trained_models.pkl",
        "final_model": models_dir / "final_model.pkl",
        "final_pipeline": models_dir / "final_pipeline.pkl",
        "evaluation_metrics": artifacts_dir / "evaluation_metrics.pkl",
        "model_comparison": artifacts_dir / "model_comparison.pkl",
        "feature_names_after_preprocessing": artifacts_dir
        / "feature_names_after_preprocessing.pkl",
        "selected_feature_names": artifacts_dir / "selected_feature_names.pkl",
        "inference_example": artifacts_dir / "inference_example.pkl",
    }

    save_pickle(metadata, artifact_paths["raw_dataset_metadata"])
    save_pickle(validation_report, artifact_paths["validation_report"])
    save_pickle(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        artifact_paths["train_test_split"],
    )
    save_pickle(label_encoder, artifact_paths["label_encoder"])
    save_pickle(label_mapping, artifact_paths["label_mapping"])
    save_pickle(fitted_preprocessor, artifact_paths["preprocessing_pipeline"])
    save_pickle(
        fitted_preprocessor.named_transformers_["num"].named_steps["imputer"],
        artifact_paths["numeric_imputer"],
    )
    save_pickle(
        fitted_preprocessor.named_transformers_["num"].named_steps["scaler"],
        artifact_paths["scaler"],
    )
    if categorical_features:
        save_pickle(
            fitted_preprocessor.named_transformers_["cat"].named_steps["imputer"],
            artifact_paths["categorical_imputer"],
        )
        save_pickle(
            fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"],
            artifact_paths["encoder"],
        )
    save_pickle(feature_selector, artifact_paths["feature_selector"])
    save_pickle(
        {name: p.named_steps["model"] for name, p in pipelines.items()},
        artifact_paths["trained_models"],
    )
    save_pickle(final_model, artifact_paths["final_model"])
    save_pickle(final_pipeline, artifact_paths["final_pipeline"])
    save_pickle(final_metrics, artifact_paths["evaluation_metrics"])
    save_pickle(comparison, artifact_paths["model_comparison"])
    save_pickle(
        all_feature_names, artifact_paths["feature_names_after_preprocessing"]
    )
    save_pickle(chosen_features, artifact_paths["selected_feature_names"])

    example_rows = X_test.head(3).copy()
    example_predictions = final_pipeline.predict(example_rows)
    inference_example = {
        "input_records": example_rows.to_dict(orient="records"),
        "predicted_labels": [
            label_mapping[int(pred)] for pred in np.asarray(example_predictions)
        ],
    }
    save_pickle(inference_example, artifact_paths["inference_example"])

    save_json(metadata, artifacts_dir / "raw_dataset_metadata.json")
    save_json(validation_report, artifacts_dir / "validation_report.json")
    save_json(final_metrics, artifacts_dir / "evaluation_metrics.json")
    save_json(comparison, artifacts_dir / "model_comparison.json")
    deploy_config = write_deployment_config(
        output_dir,
        artifact_paths,
        schema_from_features(X),
    )

    print(f"Best model: {best_name}")
    print(f"Artifacts saved to: {output_dir}")
    return {
        "best_model": best_name,
        "final_pipeline": final_pipeline,
        "metrics": final_metrics,
        "comparison": comparison,
        "artifact_paths": artifact_paths,
        "deployment_config": deploy_config,
    }


def load_final_pipeline(output_dir=None):
    output_dir_str = output_dir or CONFIG["output_dir"]
    if output_dir_str is None:
        output_dir_str = str(accelera_config.deployment_root)
    output_dir = Path(output_dir_str).resolve()
    path = output_dir / "models" / "final_pipeline.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Final pipeline not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(input_data=None, output_dir=None):
    deps = require_ml_dependencies()
    pd = deps["pd"]

    output_dir_str = output_dir or CONFIG["output_dir"]
    if output_dir_str is None:
        output_dir_str = str(accelera_config.deployment_root)
    output_dir = Path(output_dir_str).resolve()
    pipeline = load_final_pipeline(output_dir)

    if input_data is None:
        example_path = output_dir / "artifacts" / "inference_example.pkl"
        if not example_path.exists():
            raise FileNotFoundError(
                "No inference example found. Run train_model() first."
            )
        with open(example_path, "rb") as f:
            input_data = pickle.load(f)["input_records"]

    rows = pd.DataFrame(input_data)
    return pipeline.predict(rows)


def example_load_and_predict(output_dir=None):
    """Small example for loading the final pipeline and predicting new rows."""
    output_dir_str = output_dir or CONFIG["output_dir"]
    if output_dir_str is None:
        output_dir_str = str(accelera_config.deployment_root)
    output_dir = Path(output_dir_str).resolve()
    pipeline = load_final_pipeline(output_dir)
    example_path = output_dir / "artifacts" / "inference_example.pkl"
    with open(example_path, "rb") as f:
        example = pickle.load(f)

    deps = require_ml_dependencies()
    rows = deps["pd"].DataFrame(example["input_records"])
    predictions = pipeline.predict(rows)
    print("Example predictions:", predictions.tolist())
    return predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train deployment case-study models"
    )
    parser.add_argument("--dataset-path", help="Optional local CSV dataset")
    parser.add_argument("--dataset-url", help="Optional CSV URL")
    parser.add_argument("--target-column", default=CONFIG["target_column"])
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    parser.add_argument("--test-size", type=float, default=CONFIG["test_size"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        {
            "dataset_path": args.dataset_path,
            "dataset_url": args.dataset_url,
            "target_column": args.target_column,
            "output_dir": args.output_dir,
            "test_size": args.test_size,
        }
    )
    example_load_and_predict(args.output_dir)
