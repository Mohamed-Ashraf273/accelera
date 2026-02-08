import pytest
import os
import shutil
import tempfile
import pickle
import pandas as pd
from accelera.src.automl.core.testing_preprocessing import TestingPreprocessing
from sklearn.preprocessing import LabelEncoder

class TestTestingPreprocessing:
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
        with open(os.path.join(self.temp_dir, "data_columns.pkl"), "wb") as f:
            pickle.dump(data_columns, f)
        with open(os.path.join(self.temp_dir, "col_drop.pkl"), "wb") as f:
            pickle.dump(col_drop, f)
        with open(os.path.join(self.temp_dir, "target_info.pkl"), "wb") as f:
            pickle.dump(target_info, f)
        yield self.temp_dir
        shutil.rmtree(self.temp_dir)

    def test_errors_checks_testing(self):
        with pytest.raises(ValueError):
            TestingPreprocessing(df=None, folder_path=self.temp_dir)
        with pytest.raises(ValueError):
            TestingPreprocessing(df=self.df, folder_path=None)
        with pytest.raises(ValueError):
            TestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        with open(os.path.join(self.temp_dir, "target_preprocessor.pkl"), "wb") as f:
            pickle.dump({}, f)
        with pytest.raises(ValueError):
            TestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        with open(os.path.join(self.temp_dir, "training_preprocessor.pkl"), "wb") as f:
            pickle.dump({}, f)
        target_info_invalid = {
            "problem_type": "classification",
            "mode": "A",
            "median": None,
        }
        with open(os.path.join(self.temp_dir, "target_info.pkl"), "wb") as f:
            pickle.dump(target_info_invalid, f)
        with pytest.raises(ValueError):
            TestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        target_info_invalid = {
            "col_name": "target",
            "problem_type": "invalid_problem_type",
            "mode": "A",
            "median": None,
        }
        with open(os.path.join(self.temp_dir, "target_info.pkl"), "wb") as f:
            pickle.dump(target_info_invalid, f)
        with pytest.raises(ValueError):
            TestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        with pytest.raises(ValueError):
            TestingPreprocessing(df=self.df.drop(columns=["feature1"]), folder_path=self.temp_dir)

    def test_constructor(self):
        with open(os.path.join(self.temp_dir, "target_preprocessor.pkl"), "wb") as f:
            pickle.dump({}, f)
        with open(os.path.join(self.temp_dir, "training_preprocessor.pkl"), "wb") as f:
            pickle.dump({}, f)
        target_info = {
            "col_name": "target",
            "problem_type": "classification",
            "mode": "A",
            "median": None,
        }
        with open(os.path.join(self.temp_dir, "target_info.pkl"), "wb") as f:
            pickle.dump(target_info, f)

        lower_df = self.df.copy()
        for col in lower_df.select_dtypes(include="object").columns:
            lower_df[col] = lower_df[col].apply(
                lambda x: x.lower() if isinstance(x, str) else x
            )
        tp = TestingPreprocessing(
            df=self.df.drop(columns=["target"]), folder_path=self.temp_dir
        )
        assert tp.features_only is True
        assert list(tp.X_test.columns) == ["feature1", "feature2"]
        assert tp.data_columns == ["feature1", "feature2"]
        assert tp.y_test is None
        assert tp.X_test.equals(lower_df.drop(columns=["target"]))
        tp = TestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        assert tp.features_only is False
        assert list(tp.X_test.columns) == ["feature1", "feature2"]
        assert tp.data_columns == ["feature1", "feature2", "target"]
        assert tp.y_test.equals(lower_df["target"])
        assert tp.X_test.equals(lower_df.drop(columns=["target"]))

    def test_drop_columns_test(self):
        with open(os.path.join(self.temp_dir, "target_preprocessor.pkl"), "wb") as f:
            pickle.dump({}, f)
        with open(os.path.join(self.temp_dir, "training_preprocessor.pkl"), "wb") as f:
            pickle.dump({}, f)
        target_info = {
            "col_name": "target",
            "problem_type": "classification",
            "mode": "A",
            "median": None,
        }
        with open(os.path.join(self.temp_dir, "target_info.pkl"), "wb") as f:
            pickle.dump(target_info, f)

        tp = TestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        tp.drop_columns(tp.X_test, tp.col_drop)
        assert list(tp.X_test.columns) == ["feature1"]

    def test_lower_data_test(self):
        with open(os.path.join(self.temp_dir, "target_preprocessor.pkl"), "wb") as f:
            pickle.dump({}, f)
        with open(os.path.join(self.temp_dir, "training_preprocessor.pkl"), "wb") as f:
            pickle.dump({}, f)
        target_info = {
            "col_name": "target",
            "problem_type": "classification",
            "mode": "A",
            "median": None,
        }
        with open(os.path.join(self.temp_dir, "target_info.pkl"), "wb") as f:
            pickle.dump(target_info, f)

        tp = TestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        assert tp.X_test["feature1"].tolist() == ["a", "b", "c"]
        assert tp.y_test.tolist() == ["a", None, "c"]
    
    def test_target_preprocessing_test(self):
        target_df = pd.DataFrame({"target": ["a", "b", "c"]})
        label_encoder = LabelEncoder()
        label_encoder.fit(target_df["target"])
        with open(os.path.join(self.temp_dir, "target_preprocessor.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)
        with open(os.path.join(self.temp_dir, "training_preprocessor.pkl"), "wb") as f:
            pickle.dump({}, f)
        
        target_info = {
            "col_name": "target",
            "problem_type": "classification",
            "mode": "b",
            "median": None,
        }
        with open(os.path.join(self.temp_dir, "target_info.pkl"), "wb") as f:
            pickle.dump(target_info, f)

        tp = TestingPreprocessing(df=self.df, folder_path=self.temp_dir)
        tp.target_preprocessing()
        assert tp.y_test.tolist() == [0, 1, 2]
    
