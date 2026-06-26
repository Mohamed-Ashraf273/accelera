import os

from accelera.src.e2e.e2e import E2EBase


class E2E(E2EBase):
    def __init__(self):
        super().__init__()
        self.df = None

    def _run(self, content, config, graph=None):
        self.config = config
        self.graph = graph
        self.df = self._load_content(content)

        if self.graph is not None:
            return self._run_graph()

        required_keys = {"target_col", "text_col", "folder_path"}
        missing_keys = required_keys - self.config.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Missing text E2E config values: {missing}.")

        from accelera.src.auto_preprocessing.core import text_training_preprocessing

        preprocessor_class = text_training_preprocessing.TextTrainingPreprocessing
        X_train, y_train, X_test, y_test = preprocessor_class(
            self.df,
            target_col=self.config["target_col"],
            text_col=self.config["text_col"],
            folder_path=self.config["folder_path"],
            val_size=self.config.get("val_size", 0.2),
            random_state=self.config.get("random_state", 42),
            tfidf_max_features=self.config.get("tfidf_max_features", 500),
            tfidf_ngram=tuple(self.config.get("tfidf_ngram", (1, 2))),
            tfidf_max_df=self.config.get("tfidf_max_df", 0.85),
            tfidf_min_df=self.config.get("tfidf_min_df", 3),
            is_report=self.config.get("is_report", True),
        ).common_preprocessing()

        from accelera.src.accelera_automl import AutoMLClassifier

        model = AutoMLClassifier(
            time_budget=self.config.get("time_budget"),
            n_trials=self.config.get("n_trials", 50),
            cv=self.config.get("cv", 5),
            random_state=self.config.get("random_state", 42),
            n_jobs=self.config.get("n_jobs", -1),
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        model_path = self.config.get(
            "model_save_path",
            os.path.join(self.config["folder_path"], "text_model.pkl"),
        )
        self._save_model(model, model_path)
        return (predictions, y_test), model
