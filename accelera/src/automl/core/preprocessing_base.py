import os
import pickle


class PreprocessingBase:
    def __init__(self, df, folder_path=None):
        self.df = df
        self.folder_path = folder_path
        if df is None:
            raise ValueError("Dataframe cannot be None")
        if folder_path is None:
            raise ValueError("folder_path cannot be None")
        os.makedirs(self.folder_path, exist_ok=True)

    def lower_data(self):
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = self.df[col].str.lower()

    def drop_columns(self, X, col_drop):
        col_drop = list(col_drop.keys())
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
