from accelera.src.accelera_automl import AutoMLClassifier
from accelera.src.accelera_automl import AutoMLRegressor
from accelera.src.auto_preprocessing.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
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
        return (predictions, y_test), model
