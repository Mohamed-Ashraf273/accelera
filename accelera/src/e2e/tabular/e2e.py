import os
<<<<<<< HEAD
from urllib.parse import urlparse
=======
>>>>>>> ee06af6 (integrate e2e)

import pandas as pd

from accelera.src.accelera_automl import AutoMLClassifier
from accelera.src.accelera_automl import AutoMLRegressor
from accelera.src.auto_preprocessing.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
from accelera.src.config import config as acc_config
from accelera.src.e2e.e2e import E2EBase
from accelera.src.utils.dataset_retriever import retriever


class E2E(E2EBase):
    def __init__(self):
        super().__init__()
        self.df = None

    def _run(self, content, config=None, graph=None):
        self.config = config
        self.graph = graph
        self.content = content

        if self._is_google_drive_url(self.content):
            retriever.connect()
            print(f"Retrieving dataset from Google Drive URL: {self.content}")
            self.df = retriever.retrieve_dataset(
                "dataset", url=self.content, df=True
            )
            retriever.close()
        elif isinstance(self.content, pd.DataFrame):
            self.df = self.content
        else:
            raise ValueError(
                "Content must be a Google Drive URL or a pandas DataFrame."
            )

        if self.graph is not None:
            predictions, executed_graph = self.graph(self.df)
            executed_graph.save()
            path = os.path.join(acc_config.REPO_ROOT, "pipeline.pkl")
            artifcats = (predictions, executed_graph, path)
        else:
            assert self.config is not None, (
                "Config must be provided if graph is None."
            )
            X_train, y_train, X_test, y_test = ClassicalTrainingPreprocessing(
                df=self.df,
                target_col=self.config["target_col"],
                problem_type=self.config.get("problem_type", "classification"),
                folder_path=self.config.get("folder_path", None),
                val_size=self.config.get("val_size", 0.2),
                random_state=self.config.get("random_state", 42),
                cardinality_threshold=self.config.get("cardinality_threshold", 10),
                max_unique_ordinal=self.config.get("max_unique_ordinal", 10),
                missing_threshold=self.config.get("missing_threshold", 0.2),
                columns_need_to_drop=self.config.get("columns_need_to_drop", []),
                feature_importance_threshold=self.config.get(
                    "feature_importance_threshold", 0.0
                ),
            ).common_preprocessing()

            if self.config.get("problem_type", "classification") == "regression":
                model = AutoMLRegressor(
                    time_budget=self.config.get("time_budget", None),
                    n_trials=self.config.get("n_trials", 50),
                    cv=self.config.get("cv", 5),
                    random_state=self.config.get("random_state", 42),
                    n_jobs=self.config.get("n_jobs", -1),
                )
            else:
                model = AutoMLClassifier(
                    time_budget=self.config.get("time_budget", None),
                    n_trials=self.config.get("n_trials", 50),
                    cv=self.config.get("cv", 5),
                    random_state=self.config.get("random_state", 42),
                    n_jobs=self.config.get("n_jobs", -1),
                )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            self._save_model(model, self.config.get("model_save_path", "model.pkl"))
            path = os.path.join(
                acc_config.REPO_ROOT, self.config.get("model_save_path", "model.pkl")
            )
            artifcats = ((predictions, y_test), model, path)

        self._deploy(artifcats[2])
        return artifcats[0], artifcats[1]
