import pandas as pd

from accelera.src.custom.transformer import CustomTransformer


class FrequencyEncoderTransform(CustomTransformer):
    def __init__(self):
        self.mapping_ = {}
        self.cols_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.cols_ = X.columns
        for col in self.cols_:
            self.mapping_[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.cols_)

        X_copy = X.copy()
        for col in self.cols_:
            X_copy[col] = X_copy[col].map(self.mapping_[col]).fillna(0)
        return X_copy.values
