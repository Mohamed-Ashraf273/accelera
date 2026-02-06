import os
import pickle
import pandas as pd


class PreprocessingBase:
    def __init__(self, df, folder_path=None):
        self.df = df
        self.folder_path = folder_path
        if df is None:
            raise ValueError("Dataframe cannot be None")
        if df is not None and not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if folder_path is None:
            raise ValueError("folder_path cannot be None")

    def check_path_exists(self, filename):
        path = os.path.join(self.folder_path, filename)
        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist")

    def lower_data(self):
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = self.df[col].apply(
                    lambda x: x.lower() if isinstance(x, str) else x
                )

    def drop_columns(self, X, col_drop):
        col_drop = list(col_drop.keys())
        if col_drop:
            X.drop(columns=col_drop, inplace=True, errors="ignore")

    def save_pikle(self, obj, filename):
        filepath = os.path.join(self.folder_path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)

    def load_pickle(self, filename):
        filepath = os.path.join(self.folder_path, filename)
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj

    def common_preprocessing(self):
        raise NotImplementedError("Must implement common_preprocessing method.")
