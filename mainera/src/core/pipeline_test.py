import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mainera.src.core.pipeline import Pipeline


class TestPipelineCorrectness:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, self.y, self.test_data = self.sample_data()
        self.scaler = StandardScaler()
        self.scaler.fit(self.X)

    def sample_data(self):
        X, y = make_classification(
            n_samples=1000,
            n_features=25,
            n_classes=4,
            n_informative=20,
            n_redundant=3,
            n_clusters_per_class=2,
            random_state=42,
        )
        test_data = X[:50]
        return X, y, test_data

    def test_preprocessing_correctness(self):
        p = Pipeline()
        p.preprocess("scale", lambda x: x * 2.0)
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict("pred", self.test_data)
        pipeline_result = p(self.X, self.y)

        X_scaled = self.X * 2.0
        test_scaled = self.test_data * 2.0
        manual_model = LogisticRegression(random_state=42, max_iter=1000)
        manual_model.fit(X_scaled, self.y)
        manual_result = manual_model.predict(test_scaled)

        pipeline_pred = (
            pipeline_result[0]
            if isinstance(pipeline_result, list)
            else pipeline_result
        )
        assert np.array_equal(pipeline_pred, manual_result)

    def test_multiple_preprocessing_steps(self):
        p = Pipeline()
        p.preprocess("scale", lambda x: x * 2.0)
        p.preprocess("shift", lambda x: x + 1.0)
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict("pred", self.test_data)
        pipeline_result = p(self.X, self.y)

        X_step1 = self.X * 2.0
        X_step2 = X_step1 + 1.0
        test_step1 = self.test_data * 2.0
        test_step2 = test_step1 + 1.0

        manual_model = LogisticRegression(random_state=42, max_iter=1000)
        manual_model.fit(X_step2, self.y)
        manual_result = manual_model.predict(test_step2)

        pipeline_pred = (
            pipeline_result[0]
            if isinstance(pipeline_result, list)
            else pipeline_result
        )
        assert np.array_equal(pipeline_pred, manual_result)

    @pytest.mark.parametrize("parallel", [False, True])
    def test_model_prediction_correctness(self, parallel):
        p = Pipeline()
        if not parallel:
            p.disable_parallel_execution()

        p.branch(
            "preprocessing",
            p.preprocess("standard_scaler", self.scaler.transform, branch=True),
            p.preprocess(
                "normalize",
                lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8),
                branch=True,
            ),
            p.preprocess(
                "power_transform",
                lambda x: np.sign(x) * np.power(np.abs(x), 0.8),
                branch=True,
            ),
        )
        p.preprocess("clip", lambda x: np.clip(x, -5, 5))
        p.branch(
            "models",
            p.model(
                "logreg",
                LogisticRegression(random_state=42, max_iter=1000),
                branch=True,
            ),
            p.model(
                "svm",
                SVC(probability=True, random_state=42, kernel="rbf"),
                branch=True,
            ),
            p.model(
                "rf",
                RandomForestClassifier(
                    n_estimators=50, random_state=42, max_depth=10
                ),
                branch=True,
            ),
        )
        p.predict("predict", self.test_data)
        pipeline_result = p(self.X, self.y)

        def p1(x):
            return self.scaler.transform(x)

        def p2(x):
            return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        def p3(x):
            return np.sign(x) * np.power(np.abs(x), 0.8)

        def p_common(x):
            return np.clip(x, -5, 5)

        models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(probability=True, random_state=42, kernel="rbf"),
            RandomForestClassifier(
                n_estimators=50, random_state=42, max_depth=10
            ),
        ]

        preprocessors = [p1, p2, p3]

        for preproc_idx, preprocessor in enumerate(preprocessors):
            for model_idx, model in enumerate(models):
                X_processed = p_common(preprocessor(self.X))
                test_processed = p_common(preprocessor(self.test_data))

                model_clone = clone(model)
                model_clone.fit(X_processed, self.y)
                manual_result = model_clone.predict(test_processed)
                pipeline_idx = preproc_idx * 3 + model_idx
                pipeline_pred = pipeline_result[pipeline_idx]

                assert np.array_equal(pipeline_pred, manual_result), (
                    f"Mismatch for preprocessor {preproc_idx} "
                    f"with model {model_idx} (parallel={parallel})"
                )

    def test_preprocessing_model_chain_correctness(self):
        p = Pipeline()
        p.preprocess("scale", lambda x: x / 10.0)
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict("pred", self.test_data)
        pipeline_result = p(self.X, self.y)

        X_scaled = self.X / 10.0
        test_scaled = self.test_data / 10.0
        manual_model = LogisticRegression(random_state=42, max_iter=1000)
        manual_model.fit(X_scaled, self.y)
        manual_result = manual_model.predict(test_scaled)

        pipeline_pred = (
            pipeline_result[0]
            if isinstance(pipeline_result, list)
            else pipeline_result
        )
        assert np.array_equal(pipeline_pred, manual_result)
