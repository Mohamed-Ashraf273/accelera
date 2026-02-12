from accelera.src.automl.core.tabular_preprocessing_base import (
    TabularPreprocessingBase,
)
from accelera.src.automl.utils.preprocessing import lower_data

from sklearn.model_selection import train_test_split
import os
import io


class TrainingTabularPreprocessingBase(TabularPreprocessingBase):
    def __init__(self, df, target_col, val_size, random_state, folder_path=None):
        super().__init__(df, folder_path)
        self.target_col = target_col
        self.val_size = val_size
        self.random_state = random_state
        self.report_data = {}

        if self.target_col not in self.df.columns:
            raise ValueError("target_col must be one of the dataframe columns")
        if (not (isinstance(self.val_size, (int, float)))) or (
            not (0 < self.val_size < 0.5)
        ):
            raise ValueError("test size is invalid it must be less than 0.5")
        if not (self.random_state is None) and not (isinstance(self.random_state, int)):
            raise ValueError("random state is invalid it must be integer or None")

        self.target_type = self.df[self.target_col].dtype
        os.makedirs(self.folder_path, exist_ok=True)

    def data_overview(self):
        data_head = self.df.head()
        lower_data(self.df)
        lower_data_head = self.df.head()
        io_buffer = io.StringIO()
        self.df.info(buf=io_buffer)
        data_info = io_buffer.getvalue()
        numerical_df = self.df.select_dtypes(include="number")
        categorical_df = self.df.select_dtypes(include="object")
        numerical_describe, categorical_describe = None, None
        if not numerical_df.empty:
            numerical_describe = numerical_df.describe()
        if not categorical_df.empty:
            categorical_describe = categorical_df.describe()
        missing_values = self.df.isnull().sum()
        duplicates_sum = self.df.duplicated().sum()
        duplicates_percentage = self.df.duplicated().mean() * 100
        self.report_data["data_overview"] = {
            "data_head": data_head,
            "lower_data_head": lower_data_head,
            "info": data_info,
            "numerical_describe": numerical_describe,
            "categorical_describe": categorical_describe,
            "missing_values": missing_values,
            "duplicates_sum": duplicates_sum,
            "duplicates_percentage": duplicates_percentage,
            "shape": self.df.shape,
        }

    def drop_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        self.report_data["drop_duplicates"] = {
            "shape": self.df.shape,
            "duplicates_sum": self.df.duplicated().sum(),
            "duplicates_percentage": self.df.duplicated().mean() * 100,
        }

    def split_data(self):
        X, y = self.df.drop(columns=[self.target_col]), self.df[self.target_col]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state
        )
        self.report_data["split"] = {
            "val_size": self.val_size,
            "X_train_shape": X_train.shape,
            "X_val_shape": X_val.shape,
            "y_train_shape": y_train.shape,
            "y_val_shape": y_val.shape,
        }
        return X_train, X_val, y_train, y_val
