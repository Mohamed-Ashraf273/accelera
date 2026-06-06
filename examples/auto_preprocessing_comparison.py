import time
import json

from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from accelera.src.automl.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)

from accelera.src.utils.dataset_retriever import retriever


def get_data_set_info():
    with open("auto_preproceesing_ds.json", "r") as f:
        ds = json.loads(f.read())
        tabular_ds = ds["tabular_dataset"]
    return tabular_ds


def autogluon_preprocessing(
    df, label, eval_metric="f1_weighted"
):
    start_time = time.time()
    df = df.drop_duplicates()
    training_df, testing_df = train_test_split(df, test_size=0.2, random_state=42)
    predictor = TabularPredictor(
        label=label, eval_metric=eval_metric
    ).fit(training_df, time_limit=1000)
    evaluation = predictor.evaluate(testing_df)[eval_metric]
    end_time = time.time()
    total_time = end_time - start_time
    return evaluation, total_time


def without_autogluon_preprocessing(
    df,
    target_column,
    report_path,
    problem_type="classification",
    eval_metric="f1_weighted",
):
    start_time = time.time()
    training_preprocessor = ClassicalTrainingPreprocessing(
        df, target_column, problem_type, report_path,is_report=False
    )
    X_train, y_train, X_test, y_test = training_preprocessor.common_preprocessing()
    columns = [f"feature_{i}" for i in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=columns)
    X_test_df = pd.DataFrame(X_test, columns=columns)
    X_train_df[target_column] = y_train
    X_test_df[target_column] = y_test
    predictor = TabularPredictor(
        label=target_column, eval_metric=eval_metric
    ).fit(
        train_data=X_train_df,
        feature_generator=None,
        time_limit=1000,
    )
    evaluation = predictor.evaluate(X_test_df)[eval_metric]
    end_time = time.time()
    total_time = end_time - start_time
    return evaluation, total_time


def plot_comparison(results_df, problem_type, target_graph):
    plt.figure(figsize=(10, 6))
    x_range = np.arange(len(results_df["dataset"]))

    bar1=plt.bar(
        x_range-0.2,
        results_df["autogluon_" + target_graph],
        width=0.4,
        label="AutoGluon",
    )
    bar2=plt.bar(
        x_range+0.2,
        results_df["accelera_" + target_graph],
        width=0.4,
        label="Accelera",
    )
    plt.bar_label(bar1, fmt="%.2f", padding=3)
    plt.bar_label(bar2, fmt="%.2f", padding=3)
    plt.xlabel("Dataset Name")
    plt.ylabel(target_graph)
    plt.title(f"AutoGluon vs Accelera Preprocessing Comparison - {problem_type}")
    plt.legend()
    plt.xticks(x_range, results_df["dataset"], rotation=45)
    plt.tight_layout()
    plt.savefig(f"comparison_{problem_type}_{target_graph}.png")


def main():
    tabular_ds = get_data_set_info()
    for problem_type, datasets in tabular_ds.items():
        results = []
        for dataset, info in datasets.items():
            retriever.connect()
            df = retriever.retrieve_dataset(dataset, df=True)
            label = info["target_column"]
            report_path = info["report_path"]
            eval_metric = info["eval_metric"]
            autogloun_score, autogloun_time = autogluon_preprocessing(
                df, label, eval_metric=eval_metric
            )
            accelera_score, accelera_time = without_autogluon_preprocessing(
                df,
                label,
                report_path,
                problem_type=problem_type,
                eval_metric=eval_metric,
            )
            results.append(
                {
                    "dataset": dataset,
                    "dataset_shape": df.shape,
                    "autogluon_score": autogloun_score,
                    "autogluon_time": autogloun_time,
                    "accelera_score": accelera_score,
                    "accelera_time": accelera_time,
                }
            )
            retriever.close()
        results_df = pd.DataFrame(results)
        plot_comparison(results_df, problem_type,"score")
        plot_comparison(results_df, problem_type, "time")


if __name__ == "__main__":
    main()
