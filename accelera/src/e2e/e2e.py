import os
from urllib.parse import urlparse

import pandas as pd

from accelera.src.config import config as acc_config
from accelera.src.deployment.deployment import configure_deployment
from accelera.src.utils.dataset_retriever import retriever


class E2EBase:
    def __init__(self):
        self.config = None
        self.graph = None
        self.artifacts = None

    def __call__(self, content, config, graph=None):
        return self._run(content, config=config, graph=graph)

    def _is_google_drive_url(self, value: str) -> bool:
        try:
            parsed = urlparse(value.strip())
        except (AttributeError, ValueError):
            return False

        return parsed.scheme in {"http", "https"} and parsed.netloc.lower() in {
            "drive.google.com",
            "docs.google.com",
        }

    def _load_content(self, content):
        if self._is_google_drive_url(content):
            retriever.connect()
            try:
                return retriever.retrieve_dataset("dataset", url=content, df=True)
            finally:
                retriever.close()
        if isinstance(content, pd.DataFrame):
            return content.copy()
        raise ValueError("Content must be a Google Drive URL or a pandas DataFrame.")

    def _check_predict(self, X):
        if (
            "model" in self.graph.included_types
            and "predict" not in self.graph.included_types
        ):
            return self.graph.predict("predict", test_data=X)

    def _get_xy(self, df: pd.DataFrame, target: str):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")

        X = df.drop(columns=[target])
        y = df[target]
        return X, y

    def _run_graph(self):
        target = self.config.get("target_col", None)
        if target is None:
            raise ValueError("You must pass your target column in config file.")

        X, y = self._get_xy(self.df, target)
        self._check_predict(X)
        results, executed_graph = self.graph(
            X, y, select_strategy=self.config.get("select_strategy", "max")
        )
        path = os.path.join(acc_config.REPO_ROOT, "pipeline.pkl")
        executed_graph.save(path)
        self._deploy(path)
        return (results, executed_graph)

    def _save_model(self, model, path):
        import joblib

        joblib.dump(model, path)

    def _deploy(self, path: str):
        configure_deployment(path)

    def _run(self, content, config=None, graph=None):
        raise NotImplementedError(
            "This data type is not supported for Accelera E2E."
        )
