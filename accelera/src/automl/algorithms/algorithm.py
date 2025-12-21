from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from accelera.src.core.pipeline import Pipeline
from accelera.src.utils.accelera_utils import serialize
from accelera.src.wrappers.graph_report import GraphReport


class Algorithm:
    def __init__(self):
        self.pipeline = Pipeline()
        self.dumy_models_space = (
            "models",
            self.pipeline.model(
                "lr", LogisticRegression(max_iter=1000), branch=True
            ),
            self.pipeline.model(
                "rf", RandomForestClassifier(max_depth=10), branch=True
            ),
            self.pipeline.model("svc", SVC(), branch=True),
        )

        self.dumy_preprocessors_space = (
            "preprocessors",
            self.pipeline.preprocess("encode", StandardScaler(), branch=True),
        )

    def run(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
    ):
        import numpy as np

        self.pipeline.preprocess("encode", StandardScaler()).branch(
            "preprocessing",
            self.pipeline.preprocess(
                "scale",
                lambda x: np.sign(x) * np.power(np.abs(x), 0.8),
                branch=True,
            ),
            self.pipeline.model(
                "rf",
                RandomForestClassifier(max_depth=10),
                branch=True,
            ),
        ).branch(
            "models",
            self.pipeline.model(
                "lr", LogisticRegression(max_iter=1000), branch=True
            ),
            self.pipeline.model("svc", SVC(), branch=True),
        ).predict("predict", test_data=X_test).branch(
            "metrics",
            self.pipeline.metric(
                "metric",
                "f1_score",
                y_true=y_test,
                average="macro",
                branch=True,
            ),
            self.pipeline.metric(
                "metric", "accuracy_score", y_true=y_test, branch=True
            ),
        )

        predictions, best_path = self.pipeline(
            X_train, y_train, select_strategy="max"
        )

        serialize(self.pipeline, "automl.xml")
        report = GraphReport("report", "automl.xml", predictions)
        _ = report.execute()

        return best_path
