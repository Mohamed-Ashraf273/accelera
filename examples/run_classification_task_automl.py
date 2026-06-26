import csv
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.metrics import accuracy_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accelera.src.accelera_automl import AutoMLClassifier

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "accelera_automl"
RESULTS_FILE = DATA_ROOT / "run_local_data_results.csv"


def load_dataset_split(dataset_dir):
    X_train = pd.read_csv(dataset_dir / "X_train_processed.csv")
    y_train = pd.read_csv(dataset_dir / "y_train_processed.csv").iloc[:, 0]
    X_val = pd.read_csv(dataset_dir / "X_val_processed.csv")
    y_val = pd.read_csv(dataset_dir / "y_val_processed.csv").iloc[:, 0]
    return X_train, y_train, X_val, y_val


def choose_runtime_settings():
    return {"time_budget": 1500, "n_trials": 10}


def save_result(result):
    fieldnames = ["dataset", "framework", "fit_time_sec", "val_accuracy"]
    file_exists = RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def run_accelera_automl(
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
        use_ensemble=False,
        ensemble_strategy="stacked",
        stacked_include_original_features_in_meta=False,
        n_jobs=1,
        verbose=1,
    )

    started_at = perf_counter()
    model.fit(X_train, y_train)
    duration = perf_counter() - started_at
    preds = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, preds)

    print("dataset", dataset_name)
    print("fit_time_sec", round(duration, 3))
    print("best_score", model.best_score)
    print("test_accuracy", test_accuracy)
    print("leaderboard_top3", model.return_leaderboard(top_n=3))

    result = {
        "dataset": dataset_name,
        "framework": "custom_automl",
        "fit_time_sec": round(duration, 3),
        "val_accuracy": float(test_accuracy),
    }
    save_result(result)
    return result


def main():
    custom_datasets = ["titanic_preprocessing"]

    for dataset in custom_datasets:
        dataset_dir = DATA_ROOT / dataset
        if dataset_dir.exists():
            print()
            print(f"######## {dataset} ########")
            X_train, y_train, X_val, y_val = load_dataset_split(dataset_dir)
            run_accelera_automl(dataset, X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
