import os

from accelera.src.automl.core.preprocessing_base import PreprocessingBase


class TestingPreprocessing(PreprocessingBase):
    def __init__(self, df, folder_path=None):
        super().__init__(df, folder_path=folder_path)
        self.check_path_exists("data_columns.pkl")
        self.check_path_exists("col_drop.pkl")
        self.check_path_exists("target_info.pkl")
        self.check_path_exists("target_preprocessor.pkl")
        self.check_path_exists("training_preprocessor.pkl")
        self.data_columns = self.load_pickle("data_columns.pkl")
        self.col_drop = self.load_pickle("col_drop.pkl")
        self.target_info = self.load_pickle("target_info.pkl")
        self.target_preprocessor = self.load_pickle("target_preprocessor.pkl")
        self.training_preprocessor = self.load_pickle("training_preprocessor.pkl")
        self.features_only = False
        if self.target_preprocessor is None:
            raise ValueError(
                "target_preprocessor cannot be None please run training preprocessing first to create the target preprocessor"
            )
        if self.training_preprocessor is None:
            raise ValueError(
                "training_preprocessor cannot be None please run training preprocessing first to create the training preprocessor"
            )
        if self.target_info is None:
            raise ValueError(
                "target_info cannot be None please run training preprocessing first to create the target info"
            )
        if self.target_info.get("col_name") is None:
            raise ValueError(
                "target_info must contain 'col_name' key with the target column run training preprocessing first to create the target info"
            )
        if self.target_info["col_name"] not in self.data_columns:
            raise ValueError(
                "target column specified in target_info not found in data_columns run training preprocessing first to create the correct data columns"
            )
        if self.target_info.get("problem_type") is None:
            raise ValueError(
                "target_info must contain 'problem_type' key with the target problem type run training preprocessing first to create the target info"
            )
        if self.target_info["problem_type"] not in ["classification", "regression"]:
            raise ValueError(
                "problem_type in target_info must be either 'classification' or 'regression'"
            )

        if self.target_info["col_name"] not in self.df.columns:
            self.features_only = True
            self.data_columns.remove(self.target_info["col_name"])
        if not self.data_columns == list(self.df.columns):
            raise ValueError(
                "training data columns do not match the testing data columns"
            )
        self.lower_data()
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
        # 1- lower data
        # 2- drop columns
        # 3- apply feature preprocessing
        # 4- apply target preprocessing
        try:
            self.drop_columns(self.X_test, self.col_drop)
            self.X_test = self.training_preprocessor.transform(self.X_test)
            if not self.features_only:
                self.target_preprocessing()
            return self.X_test, self.y_test
        except Exception as e:
            raise ValueError(f"Error in common preprocessing: {e}")
