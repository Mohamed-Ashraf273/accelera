import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

from accelera.src.automl.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
from accelera.src.automl.core.text_training_preprocessing import (
    TextTrainingPreprocessing,
)
from accelera.src.utils.dataset_retriever import retriever


def get_data_set_info():
    with open("auto_preproceesing_ds.json", "r") as f:
        ds = json.loads(f.read())
    return ds


def autogluon_preprocessing(
    df, label, eval_metric="f1_weighted", auto_gloun_model="CAT"
):
    start_time = time.time()
    df = df.drop_duplicates()
    training_df, testing_df = train_test_split(df, test_size=0.2, random_state=42)
    predictor = TabularPredictor(label=label, eval_metric=eval_metric).fit(
        training_df,
        time_limit=1000,
        hyperparameters={auto_gloun_model: {}},
        num_bag_folds=0,
        num_stack_levels=0,
        verbosity=2,
    )
    evaluation = predictor.evaluate(testing_df)[eval_metric]
    end_time = time.time()
    total_time = end_time - start_time
    return evaluation, total_time


def handle_data_preprocessing_type(
    df,
    target_column,
    problem_type="classification",
    text_column=None,
    dataset_type="tabular_dataset",
    report_path=None,
):
    if dataset_type == "tabular_dataset":
        X_train, y_train, X_test, y_test = ClassicalTrainingPreprocessing(
            df, target_column, problem_type, folder_path=report_path, is_report=False
        ).common_preprocessing()
        columns = [f"feature_{i}" for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=columns)
        X_test_df = pd.DataFrame(X_test, columns=columns)
    elif dataset_type == "text_dataset":
        X_train, y_train, X_test, y_test = TextTrainingPreprocessing(
            df, target_column, text_column, folder_path=report_path, is_report=False
        ).common_preprocessing()
        columns = [f"feature_{i}" for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame.sparse.from_spmatrix(X_train, columns=columns)
        X_test_df = pd.DataFrame.sparse.from_spmatrix(X_test, columns=columns)

    X_train_df[target_column] = y_train
    X_test_df[target_column] = y_test
    return X_train_df, X_test_df


def without_autogluon_preprocessing(
    df,
    target_column,
    report_path,
    problem_type="classification",
    eval_metric="f1_weighted",
    ds_type="tabular_dataset",
    text_column=None,
    auto_gloun_model="CAT",
):
    start_time = time.time()
    X_train_df, X_test_df = handle_data_preprocessing_type(
        df,
        target_column,
        problem_type,
        text_column,
        dataset_type=ds_type,
        report_path=report_path,
    )

    predictor = TabularPredictor(label=target_column, eval_metric=eval_metric).fit(
        train_data=X_train_df,
        feature_generator=None,
        time_limit=1000,
        hyperparameters={auto_gloun_model: {}},
        num_bag_folds=0,
        num_stack_levels=0,
        verbosity=2,
    )
    evaluation = predictor.evaluate(X_test_df)[eval_metric]
    end_time = time.time()
    total_time = end_time - start_time
    return evaluation, total_time


def plot_comparison(
    results_df, problem_type, target_graph, ds_type="tabular_dataset"
):
    plt.figure(figsize=(10, 6))
    x_range = np.arange(len(results_df["dataset"]))

    bar1 = plt.bar(
        x_range - 0.2,
        results_df["autogluon_" + target_graph],
        width=0.4,
        label="AutoGluon",
    )
    bar2 = plt.bar(
        x_range + 0.2,
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
    plt.savefig(f"{ds_type}_comparison_{problem_type}_{target_graph}.png")


def main():
    ds = get_data_set_info()
    total_results = []
    for dataset_type, datasets_typed in ds.items():
        if dataset_type == "image_dataset":
            continue
        for problem_type, datasets in datasets_typed.items():
            results = []
            for dataset, info in datasets.items():
                retriever.connect()
                link = info["link"]
                df = retriever.retrieve_dataset(dataset, url=link, df=True)
                label = info["target_column"]
                report_path = info["report_path"]
                eval_metric = info["eval_metric"]
                auto_gloun_model = info["autoGlounModel"]
                text_column = info.get("text_column", None)
                autogloun_score, autogloun_time = autogluon_preprocessing(
                    df,
                    label,
                    eval_metric=eval_metric,
                    auto_gloun_model=auto_gloun_model,
                )
                accelera_score, accelera_time = without_autogluon_preprocessing(
                    df,
                    label,
                    report_path,
                    problem_type=problem_type,
                    eval_metric=eval_metric,
                    text_column=text_column,
                    ds_type=dataset_type,
                    auto_gloun_model=auto_gloun_model,
                )
                results.append(
                    {
                        "dataset": dataset,
                        "dataset_shape": df.shape,
                        "autogluon_score": autogloun_score,
                        "autogluon_time": autogloun_time,
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
    total_results_df.to_csv("preprocessing_comparison_results.csv", index=False)


if __name__ == "__main__":
    main()
