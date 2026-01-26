from sklearn.base import TransformerMixin, BaseEstimator


class CustomTransformsBase(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        raise NotImplementedError("must implement fit method.")

    def transform(self, X):
        raise NotImplementedError("must implement transform method.")
