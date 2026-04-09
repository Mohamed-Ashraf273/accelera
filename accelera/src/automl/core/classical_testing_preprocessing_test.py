import shutil
import tempfile

import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from accelera.src.automl.core.classical_testing_preprocessing import (
    ClassicalTestingPreprocessing,
)
from accelera.src.utils.preprocessing import drop_columns
from accelera.src.utils.preprocessing import save_pickle


class TestClassicalTestingPreprocessing:
    @pytest.fixture(autouse=True)
    def initializetion(self):
        self.temp_dir = tempfile.mkdtemp()
        data_columns = ["feature1", "feature2", "target"]
        col_drop = {"feature2": "high missing values"}
        target_info = {
            "col_name": "target",
            "problem_type": "classification",
            "mode": "A",
            "median": None,
        }
        self.df = pd.DataFrame(
            {
                "feature1": ["A", "b", "c"],
                "feature2": [None, None, None],
                "target": ["A", None, "C"],
            }
        )
        save_pickle(self.temp_dir, data_columns, "data_columns.pkl")
        save_pickle(self.temp_dir, col_drop, "col_drop.pkl")
        save_pickle(self.temp_dir, target_info, "target_info.pkl")
        yield self.temp_dir
        shutil.rmtree(self.temp_dir)

    def test_classical_errors_checks_testing(self):
        with pytest.raises(ValueError):
            ClassicalTestingPreprocessing(df=None, folder_path=self.temp_dir)
        with pytest.raises(ValueError):
            ClassicalTestingPreprocessing(df=self.df, folder_path=None)
        with pytest.raises(ValueError):
            ClassicalTestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        save_pickle(self.temp_dir, {}, "target_preprocessor.pkl")

        with pytest.raises(ValueError):
            ClassicalTestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        save_pickle(self.temp_dir, {}, "training_preprocessor.pkl")
        target_info_invalid = {
            "problem_type": "classification",
            "mode": "A",
            "median": None,
        }
        save_pickle(self.temp_dir, target_info_invalid, "target_info.pkl")
        with pytest.raises(ValueError):
            ClassicalTestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        target_info_invalid = {
            "col_name": "target",
            "problem_type": "invalid_problem_type",
            "mode": "A",
            "median": None,
        }
        save_pickle(self.temp_dir, target_info_invalid, "target_info.pkl")

        with pytest.raises(ValueError):
            ClassicalTestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        with pytest.raises(ValueError):
            ClassicalTestingPreprocessing(
                df=self.df.drop(columns=["feature1"]), folder_path=self.temp_dir
            )

    def test_classical_setup_test(self):
        save_pickle(self.temp_dir, {}, "target_preprocessor.pkl")
        save_pickle(self.temp_dir, {}, "training_preprocessor.pkl")

        target_info = {
            "col_name": "target",
            "problem_type": "classification",
            "mode": "A",
            "median": None,
        }
        save_pickle(self.temp_dir, target_info, "target_info.pkl")

        lower_df = self.df.copy()
        for col in lower_df.select_dtypes(include="object").columns:
            lower_df[col] = lower_df[col].apply(
                lambda x: x.lower() if isinstance(x, str) else x
            )
        tp = ClassicalTestingPreprocessing(
            df=self.df.drop(columns=["target"]), folder_path=self.temp_dir
        )
        assert tp.features_only is True
        assert list(tp.X_test.columns) == ["feature1", "feature2"]
        assert tp.data_columns == ["feature1", "feature2"]
        assert tp.y_test is None
        assert tp.X_test.equals(lower_df.drop(columns=["target"]))
        tp = ClassicalTestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        assert tp.features_only is False
        assert list(tp.X_test.columns) == ["feature1", "feature2"]
        assert tp.data_columns == ["feature1", "feature2", "target"]
        assert tp.y_test.equals(lower_df["target"])
        assert tp.X_test.equals(lower_df.drop(columns=["target"]))

    def test_classical_drop_columns_test(self):
        save_pickle(self.temp_dir, {}, "target_preprocessor.pkl")
        save_pickle(self.temp_dir, {}, "training_preprocessor.pkl")

        target_info = {
            "col_name": "target",
            "problem_type": "classification",
            "mode": "A",
            "median": None,
        }
        save_pickle(self.temp_dir, target_info, "target_info.pkl")

        tp = ClassicalTestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        drop_columns(tp.X_test, tp.col_drop)
        assert list(tp.X_test.columns) == ["feature1"]

    def test_classical_lower_data_test(self):
        save_pickle(self.temp_dir, {}, "target_preprocessor.pkl")
        save_pickle(self.temp_dir, {}, "training_preprocessor.pkl")

        target_info = {
            "col_name": "target",
            "problem_type": "classification",
            "mode": "A",
            "median": None,
        }
        save_pickle(self.temp_dir, target_info, "target_info.pkl")

        tp = ClassicalTestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        assert tp.X_test["feature1"].tolist() == ["a", "b", "c"]
        assert tp.y_test.tolist() == ["a", None, "c"]

    def test_classical_target_preprocessing_test(self):
        target_df = pd.DataFrame({"target": ["a", "b", "c"]})
        label_encoder = LabelEncoder()
        label_encoder.fit(target_df["target"])

        save_pickle(self.temp_dir, label_encoder, "target_preprocessor.pkl")
        save_pickle(self.temp_dir, {}, "training_preprocessor.pkl")

        target_info = {
            "col_name": "target",
            "problem_type": "classification",
            "mode": "b",
            "median": None,
        }
        save_pickle(self.temp_dir, target_info, "target_info.pkl")

        tp = ClassicalTestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        tp.target_preprocessing()
        assert tp.y_test.tolist() == [0, 1, 2]
