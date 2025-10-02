from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class CustomClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        raise NotImplementedError(
            "CustomClassifier.fit is not implemented yet."
        )

    def predict(self, X):
        raise NotImplementedError(
            "CustomClassifier.predict is not implemented yet."
        )
