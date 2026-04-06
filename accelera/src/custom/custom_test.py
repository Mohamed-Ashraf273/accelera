import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from accelera.src.accelera_pipe.core.pipeline import Pipeline
from accelera.src.custom.classifier import CustomClassifier

skip_torch = False
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset

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
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
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
except ImportError:

    class TorchDenseModel:
        pass

    skip_torch = True


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


class TestCustomModelIntegration:
    def setup_method(self):
        """Setup test data"""
        self.X = np.random.randn(100, 5)
        self.y = np.random.randint(0, 3, 100)
        self.test_data = np.random.randn(50, 5)

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

        assert pipeline_pred.shape == manual_result.shape, (
            f"Shape mismatch: pipeline {pipeline_pred.shape} "
            f"vs manual {manual_result.shape}"
        )

        assert np.array_equal(pipeline_pred, manual_result), (
            f"Predictions don't match.\n"
            f"Pipeline: {pipeline_pred[:10]}\n"
            f"Manual:   {manual_result[:10]}"
        )

    @pytest.mark.skipif(skip_torch, reason="PyTorch is not installed.")
    def test_pipeline_with_cuda_correctness(self):
        p = Pipeline()
        p.preprocess("scale", lambda x: x / 10.0)
        p.branch(
            "models",
            p.model(
                "torch_model",
                TorchDenseModel(input_dim=5, output_dim=4),
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
        p.predict("predict", self.test_data, output_func="predict")
        pipeline_result, _ = p(self.X, self.y)

        def p1(x):
            return x / 10.0

        m1 = TorchDenseModel(input_dim=5, output_dim=4)
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
