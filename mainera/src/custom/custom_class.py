import numpy as np
from classifier import CustomClassifier


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
