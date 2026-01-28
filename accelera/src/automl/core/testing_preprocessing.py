from accelera.src.automl.core.preprocessing_base import PreprocessingBase
import os


class TestingPreprocessing(PreprocessingBase):
    def __init__(self, df, folder_path=None):
        super().__init__(df, folder_path=folder_path)
        if os.path.exists(folder_path) is None:
            raise ValueError("folder_path does not exist")
        print("Loading preprocessing objects from:", folder_path)
        
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
        if self.target_info["col_name"] not in self.df.columns:
            self.features_only = True
            self.data_columns.remove(self.target_info["col_name"])
        if not self.data_columns == list(self.df.columns):
            raise ValueError(
                "training data columns do not match the testing data columns"
            )
        if self.features_only:
            self.X_test = self.df
            self.y_test = None
        else:
            self.X_test = self.df.drop(columns=[self.target_info["col_name"]])
            self.y_test = self.df[self.target_info["col_name"]]
    def target_preprocessing(self):
        if self.target_info["problem_type"] == "classification":
            self.y_test.fillna(self.target_info["mode"], inplace=True)
            self.y_test = self.target_preprocessor.transform(self.y_test)
        elif self.target_info["problem_type"] == "regression":
            self.y_test.fillna(self.target_info["median"], inplace=True)
            self.y_test = self.target_preprocessor.transform(
                self.y_test.values.reshape(-1, 1)
            ).ravel()

    def common_preprocessing(self):
        # 1- lower data
        # 2- drop columns
        # 3- apply feature preprocessing
        # 4- apply target preprocessing
        self.lower_data()
        self.drop_columns(self.X_test, self.col_drop)
        self.X_test = self.training_preprocessor.transform(self.X_test)
        if not self.features_only:
            self.target_preprocessing()
        return self.X_test, self.y_test
