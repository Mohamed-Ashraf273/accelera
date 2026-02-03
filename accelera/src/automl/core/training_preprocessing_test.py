import pytest
import pandas as pd
import tempfile
import shutil
import os
from accelera.src.automl.core.training_preprocessing import TrainingPreprocessing


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
        assert data_overview["shape"] == (17, 11)
        assert data_overview["duplicates_sum"] == 1
        assert data_overview["duplicates_percentage"] == float(1 / 17) * 100
        assert data_overview["numerical_describe"] is not None
        assert data_overview["categorical_describe"] is not None
        assert data_overview["data_head"].shape == (5, 11)
        assert data_overview["lower_data_head"].shape == (5, 11)
        assert data_overview["missing_values"].equals(
            self.df_classification.isnull().sum()
        )
