import pandas as pd

from accelera.src.custom.transformer import CustomTransformer


class DateFeatureExtractor(CustomTransformer):
    def __init__(self, cols):
        self.cols = cols
        self.feature_names_ = []
        self.median_ = {}

    def fit(self, X, y=None):
        self.feature_names_ = []
        self.median_ = {}

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.cols)
        for col in self.cols:
            self.feature_names_.extend(
                [
                    f"{col}_year",
                    f"{col}_month",
                    f"{col}_day",
                    f"{col}_weekday",
                    f"{col}_hour",
                ]
            )
            self.median_[col] = X[col].median()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.cols)
        X_output = pd.DataFrame()
        for col in self.cols:
            col_data = pd.to_datetime(X[col], errors="coerce")
            col_date = col_data.fillna(self.median_[col])
            X_output[f"{col}_year"] = col_date.dt.year.fillna(0).astype(int)
            X_output[f"{col}_month"] = col_date.dt.month.fillna(0).astype(int)
            X_output[f"{col}_day"] = col_date.dt.day.fillna(0).astype(int)
            X_output[f"{col}_weekday"] = col_date.dt.weekday.fillna(0).astype(int)
            X_output[f"{col}_hour"] = col_date.dt.hour.fillna(0).astype(int)
        return X_output.values

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_
