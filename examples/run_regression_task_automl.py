from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accelera.src.accelera_automl import AutoMLRegressor

DATASET_NAME = "sklearn_diabetes"
RESULTS_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "accelera_automl"
    / "automl_regression_results.csv"
)


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


def load_dataset_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = load_diabetes(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def save_result(result: dict[str, object]) -> None:
    fieldnames = [
        "dataset",
        "fit_time_sec",
        "val_r2",
        "val_rmse",
        "val_mae",
        "time_budget",
        "n_trials",
        "cv",
    ]
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def main() -> None:
    args = parse_args()
    X_train, X_val, y_train, y_val = load_dataset_split()

    model = AutoMLRegressor(
        time_budget=args.time_budget,
        n_trials=args.n_trials,
        cv=args.cv,
        random_state=42,
        use_ensemble=True,
        ensemble_strategy="stacked",
        stacked_include_original_features_in_meta=False,
        n_jobs=args.n_jobs,
        disable_evaluation_timeout=args.disable_evaluation_timeout,
        use_meta_learning=True,
        verbose=1,
    )

    started_at = perf_counter()
    model.fit(X_train, y_train)
    fit_time_sec = perf_counter() - started_at

    predictions = model.predict(X_val)
    val_r2 = float(r2_score(y_val, predictions))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, predictions)))
    val_mae = float(mean_absolute_error(y_val, predictions))

    print("=== Accelera AutoML Regression ===")
    print("dataset", DATASET_NAME)
    print("fit_time_sec", round(fit_time_sec, 3))
    print("best_score", model.best_score)
    print("val_r2", val_r2)
    print("val_rmse", val_rmse)
    print("val_mae", val_mae)
    print("leaderboard_top3", model.return_leaderboard(top_n=3))

    save_result(
        {
            "dataset": DATASET_NAME,
            "fit_time_sec": round(fit_time_sec, 3),
            "val_r2": val_r2,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "time_budget": args.time_budget,
            "n_trials": args.n_trials,
            "cv": args.cv,
        }
    )


if __name__ == "__main__":
    main()
