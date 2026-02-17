import pandas as pd

from accelera.src.custom.transformer import CustomTransformer


class DateFeatureExtractor(CustomTransformer):
    def __init__(self,cols):
        self.cols = cols
        self.feature_names_=[]

    def fit(self, X, y=None):
        for col in self.cols:
            self.feature_names_.extend([f"{col}_year",f"{col}_month",f"{col}_day",f"{col}_weekday",f"{col}_hour"])
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.cols)
        X_output=pd.DataFrame()
        for col in self.cols:
            median=X[col].median()
            X[col]=X[col].fillna(median)
            X_output[f"{col}_year"]=X[col].dt.year.fillna(0).astype(int)
            X_output[f"{col}_month"]=X[col].dt.month.fillna(0).astype(int)
            X_output[f"{col}_day"]=X[col].dt.day.fillna(0).astype(int)
            X_output[f"{col}_weekday"]=X[col].dt.weekday.fillna(0).astype(int)
            X_output[f"{col}_hour"]=X[col].dt.hour.fillna(0).astype(int)
        return X_output.values

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_
