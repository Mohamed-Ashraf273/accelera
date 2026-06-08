import json

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
            df, target_column, problem_type, folder_path=report_path
        ).common_preprocessing()
    elif dataset_type == "text_dataset":
        X_train, y_train, X_test, y_test = TextTrainingPreprocessing(
            df, target_column, text_column, folder_path=report_path
        ).common_preprocessing()

    return X_train, X_test, y_train, y_test


def main():
    ds = get_data_set_info()
    for dataset_type, datasets_typed in ds.items():
        if dataset_type == "image_dataset":
            continue
        for problem_type, datasets in datasets_typed.items():
            for dataset, info in datasets.items():
                retriever.connect()
                link = info["link"]
                df = retriever.retrieve_dataset(dataset, url=link, df=True)
                label = info["target_column"]
                report_path = info["report_path"]
                text_column = info.get("text_column", None)
                handle_data_preprocessing_type(
                    df, label, problem_type, text_column, dataset_type, report_path
                )
                retriever.close()


if __name__ == "__main__":
    main()
