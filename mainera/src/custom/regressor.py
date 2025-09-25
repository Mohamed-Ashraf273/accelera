from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin


class CustomRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        raise NotImplementedError("CustomRegressor.fit is not implemented yet.")

    def predict(self, X):
        raise NotImplementedError(
            "CustomRegressor.predict is not implemented yet."
        )
