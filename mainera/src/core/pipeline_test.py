import re

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mainera.src.core.pipeline import Pipeline
from mainera.src.custom.classifier import CustomClassifier
from mainera.src.custom.custom_class import CustomModel


class TestPipelineCorrectness:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, self.y, self.test_data, self.y_test = self.sample_data()
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
        y_test = y[:50]
        return X, y, test_data, y_test

    def test_preprocessing_correctness(self):
        p = Pipeline()
        p.branch(
            "preprocessing",
            p.preprocess("standard_scaler", StandardScaler(), branch=True),
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
        pipeline_result, _ = p(self.X, self.y)

        def p1(x):
            return self.scaler.transform(x)

        def p2(x):
            return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        def p3(x):
            return np.sign(x) * np.power(np.abs(x), 0.8)

        def p_common(x):
            return np.clip(x, -5, 5)

        manual_result1 = p_common(p1(self.X))
        manual_result2 = p_common(p2(self.X))
        manual_result3 = p_common(p3(self.X))

        assert np.allclose(
            pipeline_result[0], manual_result1, rtol=1e-5, atol=1e-8
        )
        assert np.allclose(
            pipeline_result[1], manual_result2, rtol=1e-5, atol=1e-8
        )
        assert np.allclose(
            pipeline_result[2], manual_result3, rtol=1e-5, atol=1e-8
        )

    def test_multiple_preprocessing_steps(self):
        p = Pipeline()
        p.preprocess("scale", lambda x: x * 2.0)
        p.preprocess("shift", lambda x: x + 1.0)
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict("pred", self.test_data)
        pipeline_result, _ = p(self.X, self.y)

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

    def test_preprocessing_model_chain_correctness(self):
        p = Pipeline()
        p.preprocess("scale", lambda x: x / 10.0)
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict("pred", self.test_data)
        pipeline_result, _ = p(self.X, self.y)

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

    @pytest.mark.parametrize(
        "parallel", [False, True], ids=["no_parallel", "parallel"]
    )
    @pytest.mark.parametrize(
        "use_predict_proba", [False, True], ids=["predict", "predict_proba"]
    )
    def test_model_prediction_correctness(self, parallel, use_predict_proba):
        p = Pipeline()
        if not parallel:
            p.disable_parallel_execution()

        p.branch(
            "preprocessing",
            p.preprocess("standard_scaler", StandardScaler(), branch=True),
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

        p.predict("predict", self.test_data, predict_proba=use_predict_proba)
        pipeline_result, _ = p(self.X, self.y)

        assert len(pipeline_result) == 9

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

                if use_predict_proba:
                    manual_result = model_clone.predict_proba(test_processed)
                else:
                    manual_result = model_clone.predict(test_processed)

                pipeline_idx = preproc_idx * 3 + model_idx
                pipeline_pred = pipeline_result[pipeline_idx]

                if use_predict_proba:
                    assert np.allclose(
                        pipeline_pred, manual_result, rtol=1e-5, atol=1e-8
                    )
                else:
                    assert np.array_equal(pipeline_pred, manual_result)

    def test_metric_node(self):
        p = Pipeline()
        p.preprocess("scale", lambda x: x * 2.0)
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict("pred", self.test_data)
        p.metric(
            "accuracy",
            "accuracy_score",
            self.y_test,
        )
        pipeline_result, _ = p(self.X, self.y)
        accuracy = pipeline_result[-1]
        X_scaled = self.X * 2.0
        test_scaled = self.test_data * 2.0
        manual_model = LogisticRegression(random_state=42, max_iter=1000)
        manual_model.fit(X_scaled, self.y)
        manual_result = manual_model.predict(test_scaled)
        manual_accuracy = np.mean(self.y_test == manual_result)
        assert accuracy["result"] == manual_accuracy
        assert accuracy["metric_name"] == "accuracy_score"

    def test_metric_erros(self):
        p = Pipeline()
        p.preprocess("scale", lambda x: x * 2.0)
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict("pred", self.test_data)
        with pytest.raises(
            ValueError, match="Metric 'accuracy' is not recognized."
        ):
            p.metric(
                "accuracy",
                "accuracy",
                self.y_test,
            )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Metric 'get_scorer' does not take (y_true, y_pred) "
                "or (y_true, y_score) or (y_true, y_prob) as arguments."
            ),
        ):
            p.metric(
                "invalid_metric",
                "get_scorer",
                self.y_test,
            )

    class CustomModel(CustomClassifier):
        def __init__(self, random_state=42):
            self.random_state = random_state
            self.centroids_ = None
            self.classes_ = None

        def fit(self, X, y):
            np.random.seed(self.random_state)
            self.classes_ = np.unique(y)

            self.centroids_ = []
            for class_label in self.classes_:
                class_points = X[y == class_label]
                centroid = np.mean(class_points, axis=0)
                self.centroids_.append(centroid)

            self.centroids_ = np.array(self.centroids_)
            return self

        def predict(self, X):
            predictions = []
            for sample in X:
                distances = []
                for centroid in self.centroids_:
                    distance = np.sqrt(np.sum((sample - centroid) ** 2))
                    distances.append(distance)

                predicted_class = self.classes_[np.argmin(distances)]
                predictions.append(predicted_class)

            return np.array(predictions)

    def test_custom_model_integration(self):
        p = Pipeline()
        p.preprocess("scale", lambda x: x / 10.0)
        p.model("custom_rf", CustomModel(random_state=42))
        p.predict("pred", self.test_data)
        pipeline_result, _ = p(self.X, self.y)

        X_scaled = self.X / 10.0
        test_scaled = self.test_data / 10.0
        manual_model = CustomModel(random_state=42)
        manual_model.fit(X_scaled, self.y)
        manual_result = manual_model.predict(test_scaled)

        pipeline_pred = (
            pipeline_result[0]
            if isinstance(pipeline_result, list)
            else pipeline_result
        )
        assert np.array_equal(pipeline_pred, manual_result)

    def test_pipeline_with_cuda_correctness(self):
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader
            from torch.utils.data import TensorDataset
        except ImportError:
            pytest.skip("PyTorch is not installed, skipping CUDA test.")

        class TorchDenseModel(CustomClassifier):
            def __init__(self, input_dim=25, output_dim=4, random_state=42):
                torch.manual_seed(random_state)
                self.device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else pytest.skip("No GPU available, skipping CUDA test.")
                )
                self.model = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_dim),
                ).to(self.device)
                self.criterion = nn.CrossEntropyLoss()
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=0.001
                )
                self.batch_size = 32
                self.epochs = 10

            def fit(self, X, y):
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
                dataset = TensorDataset(X_tensor, y_tensor)
                dataloader = DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True
                )

                self.model.train()
                for epoch in range(self.epochs):
                    for batch_X, batch_y in dataloader:
                        self.optimizer.zero_grad()
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        loss.backward()
                        self.optimizer.step()

            def predict(self, X):
                self.model.eval()
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    outputs = self.model(X_tensor)
                    _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()

        p = Pipeline()
        p.preprocess("standard_scaler", StandardScaler())
        p.branch(
            "models",
            p.model("torch_model", TorchDenseModel(), branch=True),
            p.model(
                "rf",
                RandomForestClassifier(
                    n_estimators=50, random_state=42, max_depth=10
                ),
                branch=True,
            ),
        )
        p.predict("predict", self.test_data)
        pipeline_result, _ = p(self.X, self.y)

        def p1(x):
            return self.scaler.transform(x)

        m1 = TorchDenseModel()
        m2 = RandomForestClassifier(
            n_estimators=50, random_state=42, max_depth=10
        )
        x = p1(self.X)
        m1.fit(x, self.y)
        m2.fit(x, self.y)
        r1 = m1.predict(p1(self.test_data))
        r2 = m2.predict(p1(self.test_data))
        manual_result = [r1, r2]

        assert len(pipeline_result) == len(manual_result)
        assert np.array_equal(pipeline_result[0], manual_result[0])
        assert np.array_equal(pipeline_result[1], manual_result[1])

    def test_executed_model_correctness(self):
        p = Pipeline()
        p.preprocess("scale", lambda x: x / 10.0)
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict("pred", self.test_data)
        p.metric("accuracy", "accuracy_score", y_true=self.y_test)
        pipeline_result, executed_graph = p(self.X, self.y)
        executed_graph_result = executed_graph(self.test_data)[0]

        x = self.X / 10.0
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(x, self.y)
        manual_result = model.predict(self.test_data / 10.0)
        executed_graph_result_with_metric = executed_graph(
            self.test_data, y_true=self.y_test
        )[0]

        assert pipeline_result[0] == executed_graph_result_with_metric
        assert np.array_equal(executed_graph_result, manual_result)

        executed_graph_result = executed_graph(self.test_data)[0]
        assert np.array_equal(executed_graph_result, manual_result)
