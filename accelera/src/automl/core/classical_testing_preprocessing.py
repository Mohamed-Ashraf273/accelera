from accelera.src.automl.core.testing_tabular_preprocessing_base import (
    TestingTabularPreprocessingBase,
)
from accelera.src.automl.utils.preprocessing import check_path_exists
from accelera.src.automl.utils.preprocessing import drop_columns
from accelera.src.automl.utils.preprocessing import load_pickle
from accelera.src.automl.utils.preprocessing import lower_data


class ClassicalTestingPreprocessing(TestingTabularPreprocessingBase):
    def __init__(self, df, folder_path=None):
        super().__init__(df, folder_path=folder_path)
        check_path_exists(self.folder_path, "data_columns.pkl")
        check_path_exists(self.folder_path, "col_drop.pkl")
        check_path_exists(self.folder_path, "target_info.pkl")
        self.data_columns = load_pickle(self.folder_path, "data_columns.pkl")
        self.col_drop = load_pickle(self.folder_path, "col_drop.pkl")
        self.target_info = load_pickle(self.folder_path, "target_info.pkl")

        if self.target_info is None:
            raise ValueError(
                "target_info cannot be None please run training "
                "preprocessing first to create the target info"
            )
        if self.target_info.get("col_name") is None:
            raise ValueError(
                "target_info must contain 'col_name' key with the target "
                "column run training preprocessing first to create the "
                "target info"
            )
        if self.target_info["col_name"] not in self.data_columns:
            raise ValueError(
                "target column specified in target_info not found in "
                "data_columns run training preprocessing first to create "
                "the correct data columns"
            )
        if self.target_info.get("problem_type") is None:
            raise ValueError(
                "target_info must contain 'problem_type' key with the "
                "target problem type run training preprocessing first to "
                "create the target info"
            )
        if self.target_info["problem_type"] not in [
            "classification",
            "regression",
        ]:
            raise ValueError(
                "problem_type in target_info must be either "
                "'classification' or 'regression'"
            )

        if self.target_info["col_name"] not in self.df.columns:
            self.features_only = True
            self.data_columns.remove(self.target_info["col_name"])
        if not self.data_columns == list(self.df.columns):
            raise ValueError(
                "training data columns do not match the testing data columns"
            )
        lower_data(self.df)
        if self.features_only:
            self.X_test = self.df
            self.y_test = None
        else:
            self.X_test = self.df.drop(columns=[self.target_info["col_name"]])
            self.y_test = self.df[self.target_info["col_name"]]

    def target_preprocessing(self):
        try:
            if self.target_info["problem_type"] == "classification":
                self.y_test.fillna(self.target_info["mode"], inplace=True)
                self.y_test = self.target_preprocessor.transform(self.y_test)
            elif self.target_info["problem_type"] == "regression":
                self.y_test.fillna(self.target_info["median"], inplace=True)
                self.y_test = self.target_preprocessor.transform(
                    self.y_test.values.reshape(-1, 1)
                ).ravel()
        except Exception as e:
            raise ValueError(f"Error in target preprocessing: {e}")

    def common_preprocessing(self):
        try:
            drop_columns(self.X_test, self.col_drop)
            self.X_test = self.training_preprocessor.transform(self.X_test)
            if not self.features_only:
                self.target_preprocessing()
            return self.X_test, self.y_test
        except Exception as e:
            raise ValueError(f"Error in common preprocessing: {e}")
