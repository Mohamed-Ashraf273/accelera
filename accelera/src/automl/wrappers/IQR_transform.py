import pandas as pd

from accelera.src.custom.transformer import CustomTransformer


class IQRTransform(CustomTransformer):
    def __init__(self, info, cols):
        self.info = info
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for i, col in enumerate(self.cols):
            if self.info[col]["col_type"] in ["numerical", "continuous"]:
                lower, upper = self.info[col]["outliers_info"]
                if lower is not None and upper is not None:
                    if isinstance(X_copy, pd.DataFrame):
                        X_copy[col] = X_copy[col].clip(lower, upper)
                    else:
                        X_copy[:, i] = X_copy[:, i].clip(lower, upper)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features
