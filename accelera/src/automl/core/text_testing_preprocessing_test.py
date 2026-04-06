import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from accelera.src.automl.core.text_testing_preprocesing import (
    TextTestingPreprocessing,
)
from accelera.src.utils.preprocessing import save_pickle


class TestTextTestingPreprocessing:
    @pytest.fixture(autouse=True)
    def initializetion(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_info = {
            "target_col": "class",
            "text_col": "text",
            "target_mode": "pos",
        }
        self.train = pd.DataFrame(
            {
                "text": [
                    "Hello ",
                    "It is not great",
                    "great like hello",
                    "like great hello",
                    "i dont like it",
                ],
                "class": ["pos", "neg", "pos", "pos", "neg"],
            }
        )
        self.test = pd.DataFrame(
            {
                "text": ["welcome ", "It is not bad", "i  like it"],
                "class": ["pos", None, "pos"],
            }
        )
        self.target_preprocessor = LabelEncoder()
        self.target_preprocessor.fit_transform(self.train["class"])

        yield self.temp_dir
        shutil.rmtree(self.temp_dir)

    def test_text_errors_test(self):
        with pytest.raises(ValueError):
            TextTestingPreprocessing(df=None, folder_path=self.temp_dir)

        with pytest.raises(ValueError):
            TextTestingPreprocessing(df=self.test, folder_path=None)

        with pytest.raises(ValueError):
            TextTestingPreprocessing(df=self.test, folder_path=self.temp_dir)
        save_pickle(
            self.temp_dir, self.target_preprocessor, "target_preprocessor.pkl"
        )
        with pytest.raises(ValueError):
            TextTestingPreprocessing(df=self.test, folder_path=self.temp_dir)
        save_pickle(self.temp_dir, {}, "training_preprocessor.pkl")
        with pytest.raises(ValueError):
            TextTestingPreprocessing(df=self.test, folder_path=self.temp_dir)
        self.data_info["text_col"] = "review"
        save_pickle(self.temp_dir, self.data_info, "data_info.pkl")
        with pytest.raises(ValueError):
            TextTestingPreprocessing(df=self.test, folder_path=self.temp_dir)

        self.data_info["target_col"] = "class"
        self.data_info["text_col"] = "class"
        save_pickle(self.temp_dir, self.data_info, "data_info.pkl")
        with pytest.raises(ValueError):
            TextTestingPreprocessing(df=self.test, folder_path=self.temp_dir)
        with pytest.raises(ValueError):
            tp = TextTestingPreprocessing(
                df=self.test, folder_path=self.temp_dir
            )
            tp.feature_preprocessing()
        save_pickle(self.temp_dir, {}, "target_preprocessor.pkl")
        with pytest.raises(ValueError):
            tp = TextTestingPreprocessing(
                df=self.test, folder_path=self.temp_dir
            )
            tp.target_preprocessing()

    def test_text_target_preprocessing_test(self):
        self.data_info["target_col"] = "review"
        self.data_info["text_col"] = "text"
        y = self.test["class"]
        save_pickle(self.temp_dir, self.data_info, "data_info.pkl")
        save_pickle(self.temp_dir, {}, "training_preprocessor.pkl")
        save_pickle(
            self.temp_dir, self.target_preprocessor, "target_preprocessor.pkl"
        )

        tp = TextTestingPreprocessing(self.test, self.temp_dir)
        y_test = tp.target_preprocessing()
        assert tp.features_only
        assert y_test is None

        self.data_info["target_col"] = "class"
        self.data_info["text_col"] = "text"
        self.data_info["target_mode"] = "pos"
        save_pickle(self.temp_dir, self.data_info, "data_info.pkl")
        tp = TextTestingPreprocessing(self.test, self.temp_dir)
        y_test = tp.target_preprocessing()
        assert not tp.features_only
        assert y_test is not None
        y = y.fillna("pos")
        y = self.target_preprocessor.transform(y)
        assert np.allclose(y, y_test)
        assert tp.data_info["target_mode"] == "pos"
