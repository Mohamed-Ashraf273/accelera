import pytest
import pandas as pd
import tempfile
import shutil
import os
from accelera.src.automl.core.training_preprocessing import TrainingPreprocessing
from sklearn.model_selection import train_test_split


class TestTrainingPreprocessing:
    @pytest.fixture(autouse=True)
    def temp_folder(self):
        self.temp_dir = tempfile.mkdtemp()
        yield self.temp_dir
        shutil.rmtree(self.temp_dir)

    @pytest.fixture(autouse=True)
    def sample_data(self):
        self.df_classification = pd.DataFrame(
            {
                "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                "const_feature": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                "most_nulls_feature": [
                    1,
                    2,
                    None,
                    None,
                    3,
                    None,
                    None,
                    4,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                "binary_feature": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                "ordinal_feature": [1, 2, 3, 4, 5, 6, 7, 7, 8, 7, 6, 5, 4, 3, 2, 1],
                "frequency_feature_numbers": [
                    1,
                    5,
                    4,
                    3,
                    1,
                    3,
                    4,
                    6,
                    4,
                    4,
                    5,
                    6,
                    3,
                    1,
                    5,
                    1,
                ],
                "one_hot_feature": [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                    "F",
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                    "F",
                    "A",
                    "B",
                    "C",
                    "D",
                ],
                "continuous_feature": [
                    10.5,
                    20.3,
                    30.1,
                    40.2,
                    50.5,
                    60.6,
                    70.7,
                    80.8,
                    90.9,
                    100.1,
                    110.2,
                    120.3,
                    130.4,
                    140.5,
                    150.6,
                    160.7,
                ],
                "frequency_feature": [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                    "F",
                    "G",
                    "H",
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                    "F",
                    "G",
                    "H",
                ],
                "text_feature": [
                    "This is a sample text.",
                    "Another example of text data.",
                    "Text data for testing.",
                    "Sample text for NLP tasks.",
                    "More text data here.",
                    "Text data for classification.",
                    "NLP text sample.",
                    "Text data example.",
                    "Sample text for analysis.",
                    "Another text data point.",
                    "Text data for model training.",
                    "Text data for evaluation.",
                    "Sample text input.",
                    "Text data for processing.",
                    "Another sample text.",
                    "Final text data point.",
                ],
                "Name_feature": [
                    "Alice",
                    "Bob",
                    "Charlie",
                    "David",
                    "Eve",
                    "Frank",
                    "Grace",
                    "Heidi",
                    "Ivan kan ",
                    "Judy Karl Karl",
                    "Karl Karl Karl",
                    "Liam Karl",
                    "Mia Karl",
                    "Nina",
                    "Oscar",
                    "Peggy",
                ],
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        self.df_classification = pd.concat(
            [self.df_classification, self.df_classification.iloc[[0]]],
            ignore_index=True,
        )

    def test_setup(self):
        training_preprocessing = TrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
        )
        assert training_preprocessing.df is not None
        assert training_preprocessing.target_col == "target"
        assert training_preprocessing.problem_type == "classification"
        assert os.path.exists(self.temp_dir)

    def test_setup_errors_check(self):
        with pytest.raises(ValueError):
            TrainingPreprocessing(
                df=self.df_classification,
                target_col="col1",
                problem_type="classification",
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            TrainingPreprocessing(
                df=self.df_classification,
                target_col="target",
                problem_type="Classification",
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            TrainingPreprocessing(
                df=None,
                target_col="target",
                problem_type="regression",
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            TrainingPreprocessing(
                df=self.df_classification,
                target_col="target",
                problem_type="regression",
                folder_path=None,
            )
        with pytest.raises(ValueError):
            TrainingPreprocessing(
                df=self.df_classification,
                target_col="target",
                problem_type="regression",
                folder_path=self.temp_dir,
                text_colums_name=["review"],
            )
        with pytest.raises(ValueError):
            TrainingPreprocessing(
                df=self.df_classification,
                target_col="target",
                problem_type="regression",
                folder_path=self.temp_dir,
                text_colums_name=["continuous_feature"],
            )

    def test_data_overview(self):
        training_preprocessing = TrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
        )
        training_preprocessing.data_overview()
        assert training_preprocessing.report_data is not None
        assert "data_overview" in training_preprocessing.report_data
        data_overview = training_preprocessing.report_data["data_overview"]
        assert data_overview["shape"] == (17, 12)
        assert data_overview["duplicates_sum"] == 1
        assert data_overview["duplicates_percentage"] == float(1 / 17) * 100
        assert data_overview["numerical_describe"] is not None
        assert data_overview["categorical_describe"] is not None
        assert data_overview["data_head"].shape == (5, 12)
        assert data_overview["lower_data_head"].shape == (5, 12)
        assert data_overview["missing_values"].equals(
            self.df_classification.isnull().sum()
        )

    def test_drop_dupplicates(self):
        training_preprocessing = TrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
        )
        shape_before_drop = training_preprocessing.df.shape
        training_preprocessing.data_overview()
        training_preprocessing.drop_duplicates()
        shape_after_drop = training_preprocessing.df.shape
        assert shape_after_drop[0] == shape_before_drop[0] - 1
        assert shape_after_drop[1] == shape_before_drop[1]
        assert (
            training_preprocessing.report_data["drop_duplicates"]["shape"]
            == shape_after_drop
        )
        assert self.df_classification.duplicated().sum() == 0
        assert (
            training_preprocessing.report_data["drop_duplicates"]["duplicates_sum"] == 0
        )
        assert (
            training_preprocessing.report_data["drop_duplicates"][
                "duplicates_percentage"
            ]
            == 0
        )

    def test_split_data(self):
        training_preprocessing = TrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
        )
        training_preprocessing.data_overview()
        training_preprocessing.drop_duplicates()
        X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(
            training_preprocessing.df.drop(columns=["target"]),
            training_preprocessing.df["target"],
            test_size=0.2,
            random_state=42,
        )
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        assert X_train.shape[0] == X_train_1.shape[0]
        assert X_val.shape[0] == X_val_1.shape[0]
        assert y_train.shape[0] == y_train_1.shape[0]
        assert y_val.shape[0] == y_val_1.shape[0]
        assert (
            training_preprocessing.report_data["split"]["X_train_shape"]
            == X_train.shape
        )
        assert training_preprocessing.report_data["split"]["X_val_shape"] == X_val.shape
        assert (
            training_preprocessing.report_data["split"]["y_train_shape"]
            == y_train.shape
        )
        assert training_preprocessing.report_data["split"]["y_val_shape"] == y_val.shape
        assert X_train.equals(X_train_1)
        assert X_val.equals(X_val_1)
        assert y_train.equals(y_train_1)
        assert y_val.equals(y_val_1)

    def test_get_info(self):
        training_preprocessing = TrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
            text_colums_name=["text_feature"],
        )
        training_preprocessing.data_overview()
        training_preprocessing.drop_duplicates()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        assert len(col_drop) == 4
        assert "ID" in col_drop
        assert "const_feature" in col_drop
        assert "most_nulls_feature" in col_drop
        assert col_drop["ID"] == "It is above unique_threshold 0.9"
        assert col_drop["const_feature"] == "The column is constant"
        assert col_drop["most_nulls_feature"] == "missing above missing_threshold 0.5"
        assert (
            col_drop["Name_feature"]
            == "It is above unique_threshold 0.9 and not detected as text column"
        )
        assert isinstance(info, dict)

    def test_drop_columns(self):
        training_preprocessing = TrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
            text_colums_name=["text_feature"],
        )
        training_preprocessing.data_overview()
        training_preprocessing.drop_duplicates()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        training_preprocessing.drop_col(X_train, X_val, col_drop)
        for col in col_drop.keys():
            assert col not in X_train.columns
            assert col not in X_val.columns
        assert X_train.shape[1] ==7
        assert X_val.shape[1] ==7
    def test_save_drop_columns(self):
        training_preprocessing = TrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
            text_colums_name=["text_feature"],
        )
        training_preprocessing.data_overview()
        training_preprocessing.drop_duplicates()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        training_preprocessing.save_pikle(col_drop, "col_drop.pkl")
        assert os.path.exists(os.path.join(self.temp_dir, "col_drop.pkl"))
        loaded_col_drop = training_preprocessing.load_pickle("col_drop.pkl")
        assert loaded_col_drop == col_drop
    