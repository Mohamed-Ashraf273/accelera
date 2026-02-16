import pandas as pd

from accelera.src.automl.core.preprocessing_base import PreprocessingBase


class TabularPreprocessingBase(PreprocessingBase):
    def __init__(self, df, folder_path=None):
        self.df = df
        if df is None:
            raise ValueError("Dataframe cannot be None")
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        super().__init__(folder_path)
