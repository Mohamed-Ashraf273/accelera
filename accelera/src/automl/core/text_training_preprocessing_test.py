import pytest
import shutil
import tempfile
import os
import pandas as pd
import numpy as np
from accelera.src.automl.core.text_training_preprocessing import (
    TextTrainingPreprocessing,
)
from sklearn.model_selection import train_test_split
from accelera.src.automl.utils.preprocessing import check_path_exists
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


class TestTextTrainingPreprocessing:
    @pytest.fixture(autouse=True)
    def temp_folder(self):
        self.temp_dir = tempfile.mkdtemp()
        yield self.temp_dir
        shutil.rmtree(self.temp_dir)

    @pytest.fixture(autouse=True)
    def sample_data(self):
        self.df = pd.DataFrame(
            {
                "text": [
                    "Hello ! how are you?",
                    "Welcome",
                    "Welcome",
                    "She's great",
                    "it is terrible",
                    "great terrible",
                    "terrible great welcome",
                    "Hello",
                    "Hello welcome",
                    None,
                ],
                "class": [
                    "pos",
                    "pos",
                    "pos",
                    "pos",
                    None,
                    "neg",
                    "pos",
                    "neg",
                    "pos",
                    None,
                ],
                "wrong_text": [1.1, 2, 3, 5, 5, 5, 2, 3, 4, 5],
            }
        )

    def test_text_error_checks(self):
        with pytest.raises(ValueError):
            TextTrainingPreprocessing(
                df=None, target_col="class", text_col="text", folder_path=self.temp_dir
            )
        with pytest.raises(ValueError):
            TextTrainingPreprocessing(
                df=self.df, target_col="class", text_col="text", folder_path=None
            )

        with pytest.raises(ValueError):
            TextTrainingPreprocessing(
                df=self.df,
                target_col="unvalid",
                text_col="text",
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            TextTrainingPreprocessing(
                df=self.df,
                target_col="text",
                text_col="unvalid",
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            TextTrainingPreprocessing(
                df=self.df,
                target_col="text",
                text_col="wrong_text",
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            TextTrainingPreprocessing(
                df=self.df,
                target_col="wrong_text",
                text_col="class",
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            TextTrainingPreprocessing(
                df=self.df,
                target_col="class",
                text_col="class",
                folder_path=self.temp_dir,
            )

    def test_text_setup(self):
        tp = TextTrainingPreprocessing(self.df, "class", "text", self.temp_dir)
        expected_data = self.df[["text", "class"]].copy()
        assert tp.info["target_col"] == "class"
        assert tp.info["text_col"] == "text"
        pd.testing.assert_frame_equal(tp.df, expected_data)

    def test_text_data_overview(self):
        self.df = self.df.drop(columns=["wrong_text"])
        tp = TextTrainingPreprocessing(
            df=self.df,
            target_col="class",
            text_col="text",
            folder_path=self.temp_dir,
        )
        tp.data_overview()
        assert tp.report_data is not None
        assert "data_overview" in tp.report_data
        data_overview = tp.report_data["data_overview"]
        assert data_overview["shape"] == (10, 2)
        assert data_overview["duplicates_sum"] == 1
        assert data_overview["duplicates_percentage"] == float(1 / 10) * 100
        assert data_overview["numerical_describe"] is None
        assert data_overview["categorical_describe"] is not None
        assert data_overview["data_head"].shape == (5, 2)
        assert data_overview["lower_data_head"].shape == (5, 2)
        assert data_overview["missing_values"].equals(self.df.isnull().sum())

    def test_text_drop_dupplicates(self):
        tp = TextTrainingPreprocessing(
            df=self.df,
            target_col="class",
            text_col="text",
            folder_path=self.temp_dir,
        )
        shape_before_drop = tp.df.shape
        tp.data_overview()
        tp.drop_duplicates()
        shape_after_drop = tp.df.shape
        print("shape", self.df, "shape", shape_after_drop)
        assert shape_after_drop[0] == shape_before_drop[0] - 1
        assert shape_after_drop[1] == shape_before_drop[1]
        assert tp.report_data["drop_duplicates"]["shape"] == shape_after_drop
        assert self.df.duplicated().sum() == 0
        assert tp.report_data["drop_duplicates"]["duplicates_sum"] == 0
        assert tp.report_data["drop_duplicates"]["duplicates_percentage"] == 0

    def test_text_split_data(self):
        self.df = self.df.drop(columns=["wrong_text"])

        tp = TextTrainingPreprocessing(
            df=self.df,
            target_col="class",
            text_col="text",
            folder_path=self.temp_dir,
        )
        tp.data_overview()
        tp.drop_duplicates()
        X_train_manual, X_val_manual, y_train_manual, y_val_manual = train_test_split(
            tp.df.drop(columns=["class"]),
            tp.df["class"],
            test_size=0.2,
            random_state=42,
        )
        X_train, X_val, y_train, y_val = tp.split_data()
        assert X_train.shape[0] == X_train_manual.shape[0]
        assert X_val.shape[0] == X_val_manual.shape[0]
        assert y_train.shape[0] == y_train_manual.shape[0]
        assert y_val.shape[0] == y_val_manual.shape[0]
        assert tp.report_data["split"]["X_train_shape"] == X_train.shape
        assert tp.report_data["split"]["X_val_shape"] == X_val.shape
        assert tp.report_data["split"]["y_train_shape"] == y_train.shape
        assert tp.report_data["split"]["y_val_shape"] == y_val.shape
        assert X_train.equals(X_train_manual)
        assert X_val.equals(X_val_manual)
        assert y_train.equals(y_train_manual)
        assert y_val.equals(y_val_manual)

    def test_text_make_graphs(self):
        tp = TextTrainingPreprocessing(
            df=self.df,
            target_col="class",
            text_col="text",
            folder_path=self.temp_dir,
        )
        tp.data_overview()
        tp.drop_duplicates()
        X_train, X_val, y_train, y_val = tp.split_data()
        tp.make_graphs(X_train, y_train)
        graphs_path = os.path.join(self.temp_dir, "graphs")
        assert check_path_exists(graphs_path, "")
        graphs_files = os.listdir(graphs_path)
        expected_num_graphs = 2
        assert len(graphs_files) == expected_num_graphs
        assert "class.png" in graphs_files
        assert "text.png" in graphs_files

    def test_text_custom_text_tokenizer(self):
        tp = TextTrainingPreprocessing(
            df=self.df,
            target_col="class",
            text_col="text",
            folder_path=self.temp_dir,
        )
        text = "Hello ? he's --  can’t won't 'no' history like123nice "
        tokens = tp.custom_text_tokenizer(text)
        tokens = " ".join(tokens)
        assert tokens == "hello not not no histori like 123 nice"

    def test_text_features_preprocessing(self):
        tp = TextTrainingPreprocessing(
            df=self.df,
            target_col="class",
            text_col="text",
            folder_path=self.temp_dir,
        )
        tp.data_overview()
        tp.drop_duplicates()
        tp.report_data["preprocessing"] = []
        X_train, X_val, y_train, y_val = tp.split_data()
        X_train_filled = X_train["text"].fillna("")
        X_val_filled = X_val["text"].fillna("")
        X_train_processed, X_val_processed = tp.features_preprocessing(X_train, X_val)
        training_preprocessor = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=3,
            tokenizer=tp.custom_text_tokenizer,
            token_pattern=None,
            lowercase=False,
        )

        X_train_processed_manual = training_preprocessor.fit_transform(X_train_filled)
        X_val_processed_manual = training_preprocessor.transform(X_val_filled)
        assert X_train_processed_manual.shape == X_train_processed.shape
        assert np.allclose(
            X_train_processed_manual.toarray(), X_train_processed.toarray()
        )
        assert np.allclose(X_val_processed_manual.toarray(), X_val_processed.toarray())
        assert check_path_exists(self.temp_dir, "training_preprocessor.pkl")

    def test_text_target_preprocessing(self):
        tp = TextTrainingPreprocessing(
            df=self.df,
            target_col="class",
            text_col="text",
            folder_path=self.temp_dir,
        )
        tp.data_overview()
        tp.drop_duplicates()
        tp.report_data["after_preprocessing"] = {}
        tp.report_data["preprocessing"] = []
        X_train, X_val, y_train, y_val = tp.split_data()
        mode = y_train.mode()[0]
        encoder = LabelEncoder()
        y_train_fill = y_train.fillna(mode)
        y_val_fill = y_val.fillna(mode)
        y_train_manual = encoder.fit_transform(y_train_fill)
        y_val_manual = encoder.transform(y_val_fill)
        y_train_processed, y_val_processed = tp.target_preprocessing(y_train, y_val)
        assert tp.info["target_mode"] == mode
        assert np.allclose(y_val_manual, y_val_processed)
        assert np.allclose(y_train_manual, y_train_processed)
        assert check_path_exists(self.temp_dir, "target_preprocessor.pkl")
        assert check_path_exists(self.temp_dir, "data_info.pkl")
