import re

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mainera.src.core.pipeline import Pipeline
from mainera.src.custom.classifier import CustomClassifier


class TestPipelineCorrectness:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, self.y, self.test_data, self.y_test = self.sample_data()
        self.x_unsupervised, self.y_unsupervised = make_blobs(
            n_samples=300, centers=3, n_features=5, random_state=42
        )
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
        "use_predict_proba",
        ["predict", "predict_proba"],
        ids=["predict", "predict_proba"],
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
                SVC(random_state=42, probability=True),
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

        p.predict("predict", self.test_data, output_func=use_predict_proba)
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
            SVC(random_state=42, probability=True),
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

                if use_predict_proba == "predict_proba":
                    manual_result = model_clone.predict_proba(test_processed)
                else:
                    manual_result = model_clone.predict(test_processed)

                pipeline_idx = preproc_idx * 3 + model_idx
                pipeline_pred = pipeline_result[pipeline_idx]

                if use_predict_proba == "predict_proba":
                    assert np.allclose(
                        pipeline_pred, manual_result, rtol=1e-5, atol=1e-8
                    )
                else:
                    assert np.array_equal(pipeline_pred, manual_result)

    def test_prediction_output_fun(self):
        p = Pipeline()
        p.preprocess("scale", StandardScaler())
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict(
            "pred",
            self.test_data,
            output_func="predict_proba",
            positive_class=1,
        )
        pipeline_result, _ = p(self.X, self.y)

        X_scaled = self.scaler.fit_transform(self.X)
        test_scaled = self.scaler.transform(self.test_data)
        manual_model = LogisticRegression(random_state=42, max_iter=1000)
        manual_model.fit(X_scaled, self.y)
        manual_result = manual_model.predict_proba(test_scaled)[:, 1]
        assert np.allclose(pipeline_result, manual_result, rtol=1e-5, atol=1e-8)

    def test_metric_node(self):
        p = Pipeline()
        p.preprocess("scale", lambda x: x * 2.0)
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict("pred", self.test_data, output_func="predict")
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
        assert accuracy["metric name"] == "accuracy"

    def test_metric_errors(self):
        p = Pipeline()
        p.preprocess("scale", lambda x: x * 2.0)
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict("pred", self.test_data, output_func="predict")
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
                "Metric 'get_scorer' is incompatible with "
                "the supervised or unsupervised metric structure."
            ),
        ):
            p.metric(
                "invalid_metric",
                "get_scorer",
                self.y_test,
            )

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
        p.predict("predict", self.test_data, output_func="predict")
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
        p.predict("pred", self.test_data, output_func="predict")
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

    def test_hard_voting_merge_correctness(self):
        p = Pipeline()
        p.preprocess("scale", StandardScaler())

        # Branch with multiple models
        p.branch(
            "models",
            p.model(
                "lr",
                LogisticRegression(random_state=42, max_iter=1000),
                branch=True,
            ),
            p.model(
                "rf",
                RandomForestClassifier(n_estimators=10, random_state=42),
                branch=True,
            ),
        )

        p.predict("predict", self.test_data, output_func="predict")
        p.merge("merge_node", "hard_voting")

        pipeline_result, _ = p(self.X, self.y)

        # Manual hard voting calculation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        test_scaled = scaler.transform(self.test_data)

        # Train models manually
        lr = LogisticRegression(random_state=42, max_iter=1000)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)

        lr.fit(X_scaled, self.y)
        rf.fit(X_scaled, self.y)

        # Get predictions from each model
        lr_pred = lr.predict(test_scaled)
        rf_pred = rf.predict(test_scaled)

        # Manual hard voting (majority vote)
        from scipy import stats

        predictions_stack = np.column_stack([lr_pred, rf_pred])
        manual_hard_vote = stats.mode(predictions_stack, axis=1)[0].flatten()

        # Pipeline should return the merged result
        assert len(pipeline_result) == 1, "Merge should produce single result"
        assert np.array_equal(pipeline_result[0], manual_hard_vote)

    def test_merge_with_preprocessing_branches(self):
        """Test merge with multiple preprocessing and model branches"""
        p = Pipeline()

        # Multiple preprocessing branches
        p.branch(
            "preprocessing",
            p.preprocess("scale1", StandardScaler(), branch=True),
            p.preprocess("scale2", lambda x: x * 2.0, branch=True),
        )

        # Single model applied to each preprocessing branch
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict("predict", self.test_data, output_func="predict")
        p.merge("merge_node", "hard_voting")

        pipeline_result, _ = p(self.X, self.y)

        # Manual calculation
        scaler = StandardScaler()
        X_scaled1 = scaler.fit_transform(self.X)
        test_scaled1 = scaler.transform(self.test_data)

        X_scaled2 = self.X * 2.0
        test_scaled2 = self.test_data * 2.0

        lr1 = LogisticRegression(random_state=42, max_iter=1000)
        lr2 = LogisticRegression(random_state=42, max_iter=1000)

        lr1.fit(X_scaled1, self.y)
        lr2.fit(X_scaled2, self.y)

        pred1 = lr1.predict(test_scaled1)
        pred2 = lr2.predict(test_scaled2)

        # Hard voting between the two predictions
        from scipy import stats

        predictions_stack = np.column_stack([pred1, pred2])
        manual_hard_vote = stats.mode(predictions_stack, axis=1)[0].flatten()

        assert len(pipeline_result) == 1
        assert np.array_equal(pipeline_result[0], manual_hard_vote)

    def test_merge_with_predict_proba(self):
        """Test merge with probability predictions"""
        p = Pipeline()
        p.preprocess("scale", StandardScaler())

        p.branch(
            "models",
            p.model(
                "lr",
                LogisticRegression(random_state=42, max_iter=1000),
                branch=True,
            ),
            p.model(
                "rf",
                RandomForestClassifier(n_estimators=10, random_state=42),
                branch=True,
            ),
        )

        p.predict("predict", self.test_data, output_func="predict_proba")
        p.merge("merge_node", "hard_voting")

        pipeline_result, _ = p(self.X, self.y)

        # Manual calculation with probabilities
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        test_scaled = scaler.transform(self.test_data)

        lr = LogisticRegression(random_state=42, max_iter=1000)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)

        lr.fit(X_scaled, self.y)
        rf.fit(X_scaled, self.y)

        # Get class predictions (not probabilities) for hard voting
        lr_pred = lr.predict(test_scaled)
        rf_pred = rf.predict(test_scaled)

        from scipy import stats

        predictions_stack = np.column_stack([lr_pred, rf_pred])
        manual_hard_vote = stats.mode(predictions_stack, axis=1)[0].flatten()

        assert len(pipeline_result) == 1
        assert np.array_equal(pipeline_result[0], manual_hard_vote)

    def test_merge_single_model(self):
        """Test merge behavior with only one model (edge case)"""
        p = Pipeline()
        p.preprocess("scale", StandardScaler())
        p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
        p.predict("predict", self.test_data, output_func="predict")
        p.merge("merge_node", "hard_voting")

        pipeline_result, _ = p(self.X, self.y)

        # Manual calculation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        test_scaled = scaler.transform(self.test_data)

        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_scaled, self.y)
        manual_pred = lr.predict(test_scaled)

        # With single model, merge should return the same prediction
        assert len(pipeline_result) == 1
        assert np.array_equal(pipeline_result[0], manual_pred)

    def test_executed_graph_with_merge(self):
        """Test that executed graph works correctly with merge nodes"""
        p = Pipeline()
        p.preprocess("scale", StandardScaler())

        p.branch(
            "models",
            p.model(
                "lr",
                LogisticRegression(random_state=42, max_iter=1000),
                branch=True,
            ),
            p.model(
                "rf",
                RandomForestClassifier(n_estimators=10, random_state=42),
                branch=True,
            ),
        )

        p.predict("predict", self.test_data, output_func="predict")
        p.merge("merge_node", "hard_voting")

        # First execution
        pipeline_result, executed_graph = p(self.X, self.y)

        # Second execution using executed graph
        executed_result = executed_graph(self.test_data)

        # Results should be identical
        assert len(pipeline_result) == 1
        assert len(executed_result) == 1
        assert np.array_equal(pipeline_result[0], executed_result[0])

        # Test with new data
        new_test_data = self.test_data[:10]  # Smaller subset
        new_executed_result = executed_graph(new_test_data)

        # Manual verification with new data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        new_test_scaled = scaler.transform(new_test_data)

        lr = LogisticRegression(random_state=42, max_iter=1000)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)

        lr.fit(X_scaled, self.y)
        rf.fit(X_scaled, self.y)

        lr_pred = lr.predict(new_test_scaled)
        rf_pred = rf.predict(new_test_scaled)

        from scipy import stats

        predictions_stack = np.column_stack([lr_pred, rf_pred])
        manual_hard_vote = stats.mode(predictions_stack, axis=1)[0].flatten()

        assert np.array_equal(new_executed_result[0], manual_hard_vote)

    def test_unsupervised_problem(self):
        p = Pipeline()
        p.preprocess("standard_scaler", StandardScaler())
        p.model("kmeans", KMeans(n_clusters=3, random_state=42))
        p.predict("predict", self.x_unsupervised, output_func="fit_predict")
        p.branch(
            "metric",
            p.metric(
                "silhouette", "silhouette_score", y_true=None, branch=True
            ),
            p.metric(
                "calinski_harabasz_score",
                "calinski_harabasz_score",
                y_true=None,
                branch=True,
            ),
            p.metric(
                "davies_bouldin_score",
                "davies_bouldin_score",
                y_true=None,
                branch=True,
            ),
            p.metric(
                "adjusted_rand_score",
                "adjusted_rand_score",
                y_true=self.y_unsupervised,
                branch=True,
            ),
            p.metric(
                "v_measure_score",
                "v_measure_score",
                y_true=self.y_unsupervised,
                branch=True,
            ),
        )
        pipeline_result, _ = p(self.x_unsupervised)
        kmeans = KMeans(n_clusters=3, random_state=42)
        x_scaled = self.scaler.fit_transform(self.x_unsupervised)
        labels = kmeans.fit_predict(x_scaled)
        manual_silhouette = silhouette_score(x_scaled, labels)
        manual_devies_bouldin = davies_bouldin_score(x_scaled, labels)
        manual_calinski_harabasz = calinski_harabasz_score(x_scaled, labels)
        manual_v_measure = v_measure_score(self.y_unsupervised, labels)
        manual_adjusted_rand = adjusted_rand_score(self.y_unsupervised, labels)
        assert np.allclose(pipeline_result[0]["result"], manual_silhouette)
        assert np.allclose(
            pipeline_result[1]["result"], manual_calinski_harabasz
        )
        assert np.allclose(pipeline_result[2]["result"], manual_devies_bouldin)
        assert np.allclose(pipeline_result[3]["result"], manual_adjusted_rand)
        assert np.allclose(pipeline_result[4]["result"], manual_v_measure)


class TestPipelinePathSelection:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, self.y, self.test_data, self.y_test = self.sample_data()

    def sample_data(self):
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_classes=3,
            n_informative=15,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42,
        )
        test_data = X[:50]
        y_test = y[:50]
        return X, y, test_data, y_test

    def create_branched_pipeline(self):
        p = Pipeline()

        p.branch(
            "preprocessing",
            p.preprocess("standard_scaler", StandardScaler(), branch=True),
            p.preprocess(
                "normalize",
                lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8),
                branch=True,
            ),
        )

        p.branch(
            "models",
            p.model(
                "logreg",
                LogisticRegression(random_state=42, max_iter=1000),
                branch=True,
            ),
            p.model(
                "rf",
                RandomForestClassifier(
                    n_estimators=10, random_state=42, max_depth=5
                ),
                branch=True,
            ),
        )

        p.predict("predict", self.test_data)
        p.metric("accuracy", "accuracy_score", y_true=self.y_test)

        return p

    def test_path_selection_all_strategy(self):
        p = self.create_branched_pipeline()

        pipeline_result, executed_graph = p(
            self.X, self.y, select_strategy="all"
        )

        predictions = executed_graph(self.test_data, y_true=self.y_test)

        assert len(predictions) == 4

        for i in range(len(pipeline_result)):
            assert np.array_equal(pipeline_result[i], predictions[i])

    def test_path_selection_max_strategy(self):
        p = self.create_branched_pipeline()

        pipeline_result, executed_graph = p(
            self.X, self.y, select_strategy="max"
        )

        predictions = executed_graph(self.test_data, y_true=self.y_test)

        assert len(predictions) == 1

        max_accuracy = -1
        best_prediction = None

        for i in range(len(pipeline_result)):
            accuracy = pipeline_result[i]["result"]
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_prediction = pipeline_result[i]["result"]

        assert np.array_equal(predictions[0]["result"], best_prediction)

    def test_path_selection_min_strategy(self):
        p = self.create_branched_pipeline()

        pipeline_result, executed_graph = p(
            self.X, self.y, select_strategy="min"
        )

        predictions = executed_graph(self.test_data, y_true=self.y_test)
        assert len(predictions) == 1

        min_accuracy = float("inf")
        worst_prediction = None

        for i in range(len(pipeline_result)):
            accuracy = pipeline_result[i]["result"]
            if accuracy < min_accuracy:
                min_accuracy = accuracy
                worst_prediction = pipeline_result[i]["result"]

        assert np.array_equal(predictions[0]["result"], worst_prediction)

    def test_path_selection_custom_strategy(self):
        p = self.create_branched_pipeline()

        def custom_selector(metrics):
            best_node_name = None
            best_score = -1

            for metric in metrics:
                if "node name" in metric:
                    node_name = metric["node name"]
                elif "metric name" in metric:
                    node_name = metric["metric name"]
                else:
                    node_name = str(list(metric.values())[0])

                result = metric["result"]
                score = (
                    float(result)
                    if not hasattr(result, "__iter__")
                    else float(result)
                )

                if score > best_score:
                    best_score = score
                    best_node_name = node_name

            return best_node_name

        pipeline_result, executed_graph = p(
            self.X,
            self.y,
            select_strategy="custom",
            custom_strategy=custom_selector,
        )

        predictions = executed_graph(self.test_data, y_true=self.y_test)
        assert len(predictions) == 1

        max_accuracy = -1
        best_prediction = None

        for i in range(len(pipeline_result)):
            accuracy = pipeline_result[i]["result"]
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_prediction = pipeline_result[i]["result"]

        assert np.array_equal(predictions[0]["result"], best_prediction)
