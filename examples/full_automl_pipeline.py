import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from accelera.src.accelera_automl import AutoMLClassifier
from accelera.src.accelera_automl import AutoMLRegressor
from accelera.src.auto_preprocessing.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
from accelera.src.utils.dataset_retriever import retriever

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "accelera_automl"
RESULTS_FILE = DATA_ROOT / "run_local_data_results.csv"


EXAMPLES_DIR = Path(__file__).resolve().parent


def get_data_set_info():
    with open(EXAMPLES_DIR / "auto_preprocessing_full_ds.json", "r") as f:
        ds = json.loads(f.read())
    return ds


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


def plot_comparison(results_df, problem_type, target_graph, model_name):
    plt.figure(figsize=(20, 6))
    x = np.arange(len(results_df["dataset"]))
    bar_2 = plt.bar(
        x + 0.2,
        results_df["accelera_" + target_graph],
        width=0.35,
        label="Accelera",
    )
    plt.bar_label(bar_2, fmt="%.2f", padding=3)
    plt.xticks(x, results_df["dataset"], rotation=45, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel(target_graph)
    plt.title(f"{problem_type} - AutoClean vs Accelera {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        EXAMPLES_DIR / f"full_{problem_type}_{target_graph}_{model_name}.pdf",
        format="pdf",
    )


def main():
    ds = get_data_set_info()
    total_results = []
    for dataset_type, datasets_obj in ds.items():
        if dataset_type != "tabular_dataset":
            continue
        model_name = datasets_obj["model_name"]
        datasets_problem = datasets_obj["problemType"]
        for problem_type, datasets in datasets_problem.items():
            results = []
            for dataset, info in datasets.items():
                retriever.connect()
                df = retriever.retrieve_dataset(dataset, url=info["link"], df=True)
                label = info["target_column"]
                report_path = EXAMPLES_DIR / info["report_path"]
                X_train, X_test, y_train, y_test = handle_data_preprocessing_type(
                    df, label, problem_type, dataset_type, report_path
                )
                accelera_score = handel_problem_type_model(
                    problem_type, dataset, X_train, y_train, X_test, y_test
                )

                results.append(
                    {
                        "dataset": dataset,
                        "dataset_shape": df.shape,
                        "accelera_score": accelera_score,
                        "dataset_type": dataset_type,
                        "problem_type": problem_type,
                    }
                )
                retriever.close()
            results_df = pd.DataFrame(results)
            total_results.extend(results)
            plot_comparison(results_df, problem_type, "score", model_name)

    # total_results_df = pd.DataFrame(total_results)
    # total_results_df.to_csv(
    #     EXAMPLES_DIR / "preprocessing_comparison_autoclean.csv", index=False
    # )


def choose_runtime_settings():
    return {"time_budget": 1500, "n_trials": 100}


def save_result(result):
    fieldnames = ["dataset", "framework", "fit_time_sec", "val_accuracy"]
    file_exists = RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def handel_problem_type_model(problem_type, dataset, X_train, y_train, X_val, y_val):
    if problem_type == "classification":
        run_accelera_automl_classifier(dataset, X_train, y_train, X_val, y_val)
    else:
        run_accelera_automl_regressor(dataset, X_train, y_train, X_val, y_val)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Accelera AutoML regression on sklearn's diabetes dataset."
    )
    parser.add_argument(
        "--time-budget", type=int, default=1200, help="Time budget in seconds."
    )
    parser.add_argument(
        "--n-trials", type=int, default=10, help="Maximum number of trials."
    )
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds.")
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Parallel jobs for model training."
    )
    parser.add_argument(
        "--disable-evaluation-timeout",
        action="store_true",
        help="Disable per-run evaluation timeouts.",
    )
    return parser.parse_args()


def run_accelera_automl_regressor(
    dataset_name,
    X_train,
    y_train,
    X_test,
    y_test,
):
    args = parse_args()
    model = AutoMLRegressor(
        time_budget=args.time_budget,
        n_trials=args.n_trials,
        cv=args.cv,
        random_state=42,
        use_ensemble=True,
        ensemble_strategy="stacked",
        verbose=1,
        stacked_include_original_features_in_meta=False,
        n_jobs=args.n_jobs,
        disable_evaluation_timeout=args.disable_evaluation_timeout,
        use_meta_learning=True,
    )

    started_at = perf_counter()
    model.fit(X_train, y_train)
    fit_time_sec = perf_counter() - started_at

    predictions = model.predict(X_test)
    val_r2 = float(r2_score(y_test, predictions))
    val_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    val_mae = float(mean_absolute_error(y_test, predictions))

    print("=== Accelera AutoML Regression ===")
    print("dataset", dataset_name)
    print("fit_time_sec", round(fit_time_sec, 3))
    print("best_score", model.best_score)
    print("val_r2", val_r2)
    print("val_rmse", val_rmse)
    print("val_mae", val_mae)
    print("leaderboard_top3", model.return_leaderboard(top_n=3))
    return val_r2


def run_accelera_automl_classifier(
    dataset_name,
    X_train,
    y_train,
    X_test,
    y_test,
):
    settings = choose_runtime_settings()
    model = AutoMLClassifier(
        time_budget=settings["time_budget"],
        n_trials=settings["n_trials"],
        cv=3,
        random_state=42,
        use_ensemble=True,
        ensemble_strategy="stacked",
        verbose=1,
        stacked_include_original_features_in_meta=False,
        n_jobs=1,
    )

    started_at = perf_counter()
    model.fit(X_train, y_train)
    duration = perf_counter() - started_at
    preds = model.predict(X_test)
    test_accuracy = f1_score(y_test, preds, average="macro")

    print("dataset", dataset_name)
    print("fit_time_sec", round(duration, 3))
    print("best_score", model.best_score)
    print("test_accuracy", test_accuracy)
    print("leaderboard_top3", model.return_leaderboard(top_n=3))
    return test_accuracy


if __name__ == "__main__":
    main()
