import numpy as np

from mainera.src.core.pipeline import Pipeline
from mainera.src.custom.classifier import CustomClassifier


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
            pipeline_result[0] if isinstance(pipeline_result, list) else pipeline_result
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
