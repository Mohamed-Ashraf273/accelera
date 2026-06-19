import json
from pathlib import Path

from accelera.src.auto_preprocessing.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
from accelera.src.auto_preprocessing.core.classification_image_training_preprocessing import (  # noqa: E501
    ClassificationImageTrainingPreprocessing,
)
from accelera.src.auto_preprocessing.core.text_training_preprocessing import (
    TextTrainingPreprocessing,
)
from accelera.src.utils.dataset_retriever import retriever

EXAMPLES_DIR = Path(__file__).resolve().parent

# Demo 8
print("=" * 80)
print("===Accelera Auto Preprocessing Demo: For Tabular Data , Text and Image===")
print(
    f"File running: {EXAMPLES_DIR}/auto_preprocessing_demo.py\n"
    f"This file runs datasets are existing in {EXAMPLES_DIR}/auto_preprocessing_ds\n"
    "The goal is this demo to run auto preprocessing for each "
    "type of data and generate folder for each data has pkl files and report"
)
print("=" * 80)


def get_data_set_info():
    with open(EXAMPLES_DIR / "auto_preprocessing_ds.json", "r") as f:
        ds = json.loads(f.read())
        print(f"Load Datasets from {EXAMPLES_DIR}/auto_preproceesing_ds json file")
    return ds


def handle_data_preprocessing_tabular_text(
    df,
    target_column,
    problem_type="classification",
    text_column=None,
    dataset_type="tabular_dataset",
    report_path=None,
    columns_need_to_drop=[],
):
    if dataset_type == "tabular_dataset":
        print("Starting tabular Preprocessing")

        X_train, y_train, X_test, y_test = ClassicalTrainingPreprocessing(
            df,
            target_column,
            problem_type,
            folder_path=report_path,
            columns_need_to_drop=columns_need_to_drop,
        ).common_preprocessing()
    elif dataset_type == "text_dataset":
        print("Starting text Preprocessing")
        X_train, y_train, X_test, y_test = TextTrainingPreprocessing(
            df, target_column, text_column, folder_path=report_path
        ).common_preprocessing()
    print(
        f"\nFinish Preprocessing find the report in "
        f"{report_path} and return X_train, X_test, y_train, y_test"
    )
    return X_train, X_test, y_train, y_test


def classifcation_problem(training_folder_images, folder_path, augment, image_size):
    print("\nStarting Classification Image Preprocessing")
    augment = augment == "True"
    training_preprocessor = ClassificationImageTrainingPreprocessing(
        training_folder_images=training_folder_images,
        folder_path=folder_path,
        validation_folder_images=None,
        split_training=True,
        val_size=0.2,
        images_size=image_size,
        augment=augment,
    )
    training_loader, validation_loader = training_preprocessor.common_preprocessing()
    print(
        f"Finish Preprocessing find the "
        f"report in {folder_path} training_loader, validation_loader"
    )
    return training_loader, validation_loader


def main():
    ds = get_data_set_info()
    for dataset_type, datasets_obj in ds.items():
        print(f"\nNow Process {dataset_type} datasets")
        if dataset_type == "image_dataset":
            for problem_type, datasets in datasets_obj.items():
                for ds_name, ds_info in datasets.items():
                    print(f"\nDataset Name {ds_name}")
                    image_size = (
                        ds_info["image_size"]["width"],
                        ds_info["image_size"]["height"],
                    )
                    if problem_type == "classification":
                        classifcation_problem(
                            EXAMPLES_DIR / ds_info["train_folder"],
                            EXAMPLES_DIR / ds_info["report_path"],
                            ds_info["augment"],
                            image_size=image_size,
                        )

        else:
            datasets_problem = datasets_obj["problemType"]
            for problem_type, datasets in datasets_problem.items():
                for dataset, info in datasets.items():
                    print(f"\ndataset name {dataset}")
                    retriever.connect()
                    link = info["link"]
                    print(
                        "\nDownload dataset from google drive using the given link"
                    )
                    df = retriever.retrieve_dataset(dataset, url=link, df=True)
                    label = info["target_column"]
                    report_path = EXAMPLES_DIR / info["report_path"]
                    text_column = info.get("text_column", None)
                    columns_need_to_drop = info.get("columns_need_to_drop", [])

                    handle_data_preprocessing_tabular_text(
                        df,
                        label,
                        problem_type,
                        text_column,
                        dataset_type,
                        report_path,
                        columns_need_to_drop,
                    )
                    retriever.close()


if __name__ == "__main__":
    main()
