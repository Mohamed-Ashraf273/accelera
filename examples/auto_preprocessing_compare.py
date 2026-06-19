import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autocleanml import AutoCleanML
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from accelera.src.automl.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
from accelera.src.utils.dataset_retriever import retriever

EXAMPLES_DIR = Path(__file__).resolve().parent


def get_data_set_info():
    with open(EXAMPLES_DIR / "auto_preprocessing_full_ds.json", "r") as f:
        ds = json.loads(f.read())
    return ds


def get_compared_models(problem_type):
    if problem_type == "classification":
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(random_state=42),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
        }
    else:
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state=42),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "KNN": KNeighborsRegressor(),
        }


def models_evaluation(
    X_train, X_test, y_train, y_test, problem_type="classification"
):
    models_score = {}
    for name, model in get_compared_models(problem_type).items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if problem_type == "classification":
            score = f1_score(y_test, preds, average="micro")
        else:
            score = r2_score(y_test, preds)
        models_score[name] = score
    return np.mean(list(models_score.values()))


def auto_clean_preprocessing(df, label, problem_type="classification"):
    start_time = time.time()
    df = df.drop_duplicates()
    cleaner = AutoCleanML(target=label)
    X_train, X_test, y_train, y_test, _ = cleaner.fit_transform(df)
    evaluation = models_evaluation(X_train, X_test, y_train, y_test, problem_type)
    end_time = time.time()
    return evaluation, end_time - start_time


def handle_data_preprocessing_type(
    df,
    target_column,
    problem_type="classification",
    dataset_type="tabular_dataset",
    report_path=None,
):
    if dataset_type == "tabular_dataset":
        X_train, y_train, X_test, y_test = ClassicalTrainingPreprocessing(
            df, target_column, problem_type, folder_path=report_path, is_report=False
        ).common_preprocessing()

    return X_train, X_test, y_train, y_test


def accelera_preprocessing(
    df,
    target_column,
    report_path,
    problem_type="classification",
    ds_type="tabular_dataset",
):
    start_time = time.time()
    X_train_df, X_test_df, y_train, y_test = handle_data_preprocessing_type(
        df,
        target_column,
        problem_type,
        dataset_type=ds_type,
        report_path=report_path,
    )

    evaluation = models_evaluation(
        X_train_df, X_test_df, y_train, y_test, problem_type
    )
    end_time = time.time()
    return evaluation, end_time - start_time


def plot_comparison(
    results_df, problem_type, target_graph, ds_type="tabular_dataset"
):
    plt.figure(figsize=(20, 6))
    x = np.arange(len(results_df["dataset"]))
    bar_1 = plt.bar(
        x - 0.2,
        results_df["autoclean_" + target_graph],
        width=0.35,
        label="Autoclean",
    )
    bar_2 = plt.bar(
        x + 0.2,
        results_df["accelera_" + target_graph],
        width=0.35,
        label="Accelera",
    )
    plt.bar_label(bar_1, fmt="%.2f", padding=3)
    plt.bar_label(bar_2, fmt="%.2f", padding=3)
    plt.xticks(x, results_df["dataset"], rotation=45, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel(target_graph)
    plt.title(f"{problem_type} - AutoClean vs Accelera")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        EXAMPLES_DIR / f"{ds_type}_comparison_{problem_type}_{target_graph}.png"
    )


def main():
    ds = get_data_set_info()
    total_results = []
    for dataset_type, datasets_obj in ds.items():
        if dataset_type != "tabular_dataset":
            continue
        datasets_problem = datasets_obj["problemType"]
        for problem_type, datasets in datasets_problem.items():
            results = []
            for dataset, info in datasets.items():
                retriever.connect()
                df = retriever.retrieve_dataset(dataset, url=info["link"], df=True)
                label = info["target_column"]
                report_path = info["report_path"]
                autoclean_score, autoclean_time = auto_clean_preprocessing(
                    df,
                    label,
                    problem_type=problem_type,
                )
                accelera_score, accelera_time = accelera_preprocessing(
                    df,
                    label,
                    report_path,
                    problem_type=problem_type,
                    ds_type=dataset_type,
                )
                results.append(
                    {
                        "dataset": dataset,
                        "dataset_shape": df.shape,
                        "autoclean_score": autoclean_score,
                        "autoclean_time": autoclean_time,
                        "accelera_score": accelera_score,
                        "accelera_time": accelera_time,
                        "dataset_type": dataset_type,
                        "problem_type": problem_type,
                    }
                )
                retriever.close()
            results_df = pd.DataFrame(results)
            total_results.extend(results)
            plot_comparison(results_df, problem_type, "score", dataset_type)
            plot_comparison(results_df, problem_type, "time", dataset_type)

    total_results_df = pd.DataFrame(total_results)
    total_results_df.to_csv(
        EXAMPLES_DIR / "preprocessing_comparison_autoclean.csv", index=False
    )


if __name__ == "__main__":
    main()
