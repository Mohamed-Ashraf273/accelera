from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin


class CustomClusterer(BaseEstimator, ClusterMixin):
    def fit(self, X, y=None):
        raise NotImplementedError("CustomClusterer.fit is not implemented yet.")

    def predict(self, X):
        raise NotImplementedError("CustomClusterer.predict is not implemented yet.")

    def fit_predict(self, X, y=None):
        raise NotImplementedError(
            "CustomClusterer.fit_predict is not implemented yet."
        )
