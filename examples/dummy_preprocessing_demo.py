import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from accelera.src.automl.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
from accelera.src.config import config
from accelera.src.utils.dataset_retriever import retriever

output_dir = config.DUMMY_PREPROCESSING_OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)


def get_data_set_info():
    with open("auto_preproceesing_ds.json", "r") as f:
        return json.loads(f.read())


def handel_data_raw(df, target_col):
    df = df.drop_duplicates().copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    X_train[cat_features] = X_train[cat_features].fillna("missing")
    X_test[cat_features] = X_test[cat_features].fillna("missing")

    return X_train, X_test, y_train, y_test, cat_features


def handle_data_preprocessing_type(
    df,
    target_column,
    dataset_name,
    problem_type="classification",
    dataset_type="tabular_dataset",
    report_path=None,
    columns_need_to_drop=None,
):
    if columns_need_to_drop is None:
        columns_need_to_drop = []

    if dataset_type != "tabular_dataset":
        return None

    if problem_type == "classification":
        model_raw = CatBoostClassifier(iterations=200, verbose=False)
        model_pre = CatBoostClassifier(iterations=200, verbose=False)
    else:
        model_raw = CatBoostRegressor(iterations=200, verbose=False)
        model_pre = CatBoostRegressor(iterations=200, verbose=False)

    start = time.time()

    X_train, X_test, y_train, y_test, cat_features = handel_data_raw(
        df, target_column
    )

    model_raw.fit(X_train, y_train, cat_features=cat_features)
    y_pred = model_raw.predict(X_test)

    if problem_type == "classification":
        raw_score = f1_score(y_test, y_pred, average="weighted")
    else:
        raw_score = r2_score(y_test, y_pred)

    raw_time = time.time() - start

    start = time.time()

    X_train_p, y_train_p, X_test_p, y_test_p = ClassicalTrainingPreprocessing(
        df,
        target_column,
        problem_type,
        folder_path=report_path,
        columns_need_to_drop=columns_need_to_drop,
        is_report=False,
    ).common_preprocessing()

    model_pre.fit(X_train_p, y_train_p)
    y_pred_p = model_pre.predict(X_test_p)

    if problem_type == "classification":
        pre_score = f1_score(y_test_p, y_pred_p, average="weighted")
    else:
        pre_score = r2_score(y_test_p, y_pred_p)

    pre_time = time.time() - start

    return {
        "dataset": dataset_name,
        "raw_score": raw_score,
        "raw_time": raw_time,
        "pre_score": pre_score,
        "pre_time": pre_time,
    }


def plot_and_save(df):
    x = np.arange(len(df))

    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - 0.1, df["raw_time"], width=0.2, label="Raw")
    b2 = ax.bar(x + 0.1, df["pre_time"], width=0.2, label="Preprocessed")

    ax.bar_label(b1, fmt="%.2f")
    ax.bar_label(b2, fmt="%.2f")

    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=45)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training Time Comparison")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - 0.2, df["raw_score"], width=0.4, label="Raw")
    b2 = ax.bar(x + 0.2, df["pre_score"], width=0.4, label="Preprocessed")

    ax.bar_label(b1, fmt="%.3f")
    ax.bar_label(b2, fmt="%.3f")

    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=45)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_comparison.png"))
    plt.close()


def main():
    ds = get_data_set_info()
    results = []

    for dataset_type, datasets_typed in ds.items():
        if dataset_type == "image_dataset":
            continue

        for problem_type, datasets in datasets_typed.items():
            for dataset, info in datasets.items():
                retriever.connect()
                df = retriever.retrieve_dataset(dataset, url=info["link"], df=True)

                result = handle_data_preprocessing_type(
                    df=df,
                    target_column=info["target_column"],
                    dataset_name=dataset,
                    problem_type=problem_type,
                    dataset_type=dataset_type,
                    report_path=info["report_path"],
                    columns_need_to_drop=info.get("columns_need_to_drop", []),
                )

                if result:
                    results.append(result)

                retriever.close()

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "final_results.csv"), index=False)

    print(df_results)

    plot_and_save(df_results)


if __name__ == "__main__":
    main()
