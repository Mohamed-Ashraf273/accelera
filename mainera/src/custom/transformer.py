from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        raise NotImplementedError("CustomTransformer.fit is not implemented yet.")

    def transform(self, X):
        raise NotImplementedError("CustomTransformer.transform is not implemented yet.")
