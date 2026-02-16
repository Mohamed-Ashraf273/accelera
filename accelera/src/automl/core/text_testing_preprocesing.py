from accelera.src.automl.core.testing_tabular_preprocessing_base import (
    TestingTabularPreprocessingBase,
)
from accelera.src.automl.utils.preprocessing import check_path_exists
from accelera.src.automl.utils.preprocessing import load_pickle
from accelera.src.automl.utils.preprocessing import lower_data


class TextTestingPreprocessing(TestingTabularPreprocessingBase):
    def __init__(self, df, folder_path=None):
        super().__init__(df, folder_path=folder_path)
        check_path_exists(self.folder_path, "data_info.pkl")
        self.data_info = load_pickle(self.folder_path, "data_info.pkl")
        self.target_col = self.data_info["target_col"]
        self.text_col = self.data_info["text_col"]
        self.target_mode = self.data_info["target_mode"]
        if self.text_col == self.target_col:
            raise ValueError(
                "target column and text column must not be the same"
            )
        if self.text_col not in self.df.columns:
            raise ValueError(f"data dose not has this text col {self.text_col}")
        if self.target_col not in self.df.columns:
            self.features_only = True

    def feature_preprocessing(self):
        try:
            X_test = self.df[self.text_col].fillna("")
            X_test = self.training_preprocessor.transform(X_test)
            return X_test
        except Exception as e:
            raise ValueError(f"Error in target preprocessing: {e}")

    def target_preprocessing(self):
        try:
            if self.features_only:
                return None
            else:
                y_test = self.df[self.target_col].fillna(self.target_mode)
                y_test = self.target_preprocessor.transform(y_test)
            return y_test
        except Exception as e:
            raise ValueError(f"Error in target preprocessing: {e}")

    def common_preprocessing(self):
        lower_data(self.df)
        X_test = self.feature_preprocessing()
        y_test = self.target_preprocessing()
        return X_test, y_test
