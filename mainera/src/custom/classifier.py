from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=42):
        super().__init__()
        self.random_state = random_state
        self.centroids_ = None
        self.classes_ = None
        self.n_features_in_ = None
    def fit(self, X, y):
        raise NotImplementedError(
            "CustomClassifier.fit is not implemented yet."
        )

    def predict(self, X):
        raise NotImplementedError(
            "CustomClassifier.predict is not implemented yet."
        )
