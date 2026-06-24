import json
from pathlib import Path

from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from accelera.src.auto_preprocessing.core.text_training_preprocessing import (
    TextTrainingPreprocessing,
)
from accelera.src.utils.dataset_retriever import retriever

EXAMPLES_DIR = Path(__file__).resolve().parent


def get_data_set_info():
    with open(EXAMPLES_DIR / "auto_preprocessing_full_ds.json", "r") as f:
        ds = json.loads(f.read())
    return ds


def handel_data_model(df, label, text_col, report_path):
    X_train, y_train, X_test, y_test = TextTrainingPreprocessing(
        df, label, text_col, folder_path=report_path, is_report=False
    ).common_preprocessing()
    model = XGBClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    evaluation = f1_score(y_test, prediction, average="macro")
    print("evaluation f1 score")
    print(evaluation)


def main():
    ds = get_data_set_info()
    for dataset_type, datasets_obj in ds.items():
        if dataset_type != "text_dataset":
            continue
        datasets_problem = datasets_obj["problemType"]
        for _, datasets in datasets_problem.items():
            for dataset, info in datasets.items():
                retriever.connect()
                df = retriever.retrieve_dataset(dataset, url=info["link"], df=True)
                label = info["target_column"]
                report_path = info["report_path"]
                text_col = info["text_column"]
                handel_data_model(df, label, text_col, report_path)


if __name__ == "__main__":
    main()
