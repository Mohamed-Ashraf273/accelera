import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from accelera.src.automl.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
from accelera.src.automl.wrappers.frequency_encoder_transform import (
    FrequencyEncoderTransform,
)
from accelera.src.automl.wrappers.IQR_transform import IQRTransform
from accelera.src.automl.utils.preprocessing import (
    load_pickle,
    check_path_exists,
)


class TestClassicalTrainingPreprocessing:
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
                "const_feature": [
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                ],
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
                "binary_feature": [
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                ],
                "ordinal_feature": [
                    1,
                    2,
                    3,
                    4,
                    1,
                    2,
                    3,
                    4,
                    1,
                    2,
                    3,
                    4,
                    4,
                    3,
                    2,
                    1,
                ],
                "one_hot_feature": [
                    "A",
                    "B",
                    "C",
                    "A",
                    "B",
                    "C",
                    "A",
                    "B",
                    "C",
                    "A",
                    "B",
                    "C",
                    "A",
                    "B",
                    "C",
                    "A",
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
                    "J",
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

    def test_classical_training_preprocessing_setup(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
        )
        assert training_preprocessing.df is not None
        assert training_preprocessing.target_col == "target"
        assert training_preprocessing.problem_type == "classification"
        assert check_path_exists(self.temp_dir, "")

    def test_classical_errors_check(self):
        with pytest.raises(ValueError):
            ClassicalTrainingPreprocessing(
                df=self.df_classification,
                target_col="col1",
                problem_type="classification",
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            ClassicalTrainingPreprocessing(
                df=self.df_classification,
                target_col="target",
                problem_type="Crassification",
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            ClassicalTrainingPreprocessing(
                df=None,
                target_col="target",
                problem_type="regression",
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            ClassicalTrainingPreprocessing(
                df=self.df_classification,
                target_col="target",
                problem_type="regression",
                folder_path=self.temp_dir,
            )
        with pytest.raises(ValueError):
            ClassicalTrainingPreprocessing(
                df=self.df_classification,
                target_col="target",
                problem_type="regression",
                folder_path=None,
            )
        with pytest.raises(ValueError):
            ClassicalTrainingPreprocessing(
                df=self.df_classification,
                target_col="target",
                problem_type=None,
                folder_path=self.temp_dir,
            )

    def test_classical_data_overview(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
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

    def test_classical_drop_dupplicates(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
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

    def test_classical_split_data(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
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

    def test_classical_get_info(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
        )
        training_preprocessing.data_overview()
        training_preprocessing.drop_duplicates()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        assert len(col_drop) == 5
        assert "ID" in col_drop
        assert "const_feature" in col_drop
        assert "most_nulls_feature" in col_drop
        assert "text_feature" in col_drop
        assert col_drop["ID"] == "It is above unique_threshold 0.9"
        assert col_drop["const_feature"] == "The column is constant"
        assert col_drop["most_nulls_feature"] == "Missing above missing_threshold 0.5"
        assert col_drop["Name_feature"] == "It is above unique_threshold 0.9"
        assert col_drop["text_feature"] == "It is above unique_threshold 0.9"
        assert isinstance(info, dict)

    def test_classical_drop_columns(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
        )
        training_preprocessing.data_overview()
        training_preprocessing.drop_duplicates()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        training_preprocessing.drop_col(X_train, X_val, col_drop)
        for col in col_drop.keys():
            assert col not in X_train.columns
            assert col not in X_val.columns
        assert X_train.shape[1] == 5
        assert X_val.shape[1] == 5
        assert check_path_exists(self.temp_dir, "col_drop.pkl")
        loaded_col_drop = load_pickle(self.temp_dir, "col_drop.pkl")
        assert loaded_col_drop == col_drop

    def test_classical_detect_column_types(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
            cardinality_threshold=6,
            max_unique_ordinal=8,
        )
        training_preprocessing.data_overview()
        training_preprocessing.drop_duplicates()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        training_preprocessing.drop_col(X_train, X_val, col_drop)
        (
            binary_cols,
            numerical_cols,
            one_hot_cols,
            frequency_cols,
            ordinal_cols,
            _,
        ) = training_preprocessing.detect_column_types(X_train, info)
        assert set(binary_cols) == {"binary_feature"}
        assert set(one_hot_cols) == {"one_hot_feature"}
        assert set(frequency_cols) == {
            "frequency_feature",
        }
        assert set(numerical_cols) == {"continuous_feature"}
        assert set(ordinal_cols) == {"ordinal_feature"}

    def test_classical_make_graphs(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
            cardinality_threshold=6,
            max_unique_ordinal=8,
        )
        training_preprocessing.data_overview()
        training_preprocessing.drop_duplicates()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        training_preprocessing.drop_col(X_train, X_val, col_drop)
        (
            binary_cols,
            numerical_cols,
            one_hot_cols,
            frequency_cols,
            ordinal_cols,
            _,
        ) = training_preprocessing.detect_column_types(X_train, info)
        training_preprocessing.make_graphs(X_train, y_train, info)
        graphs_path = os.path.join(self.temp_dir, "graphs")
        assert check_path_exists(graphs_path, "")
        graphs_files = os.listdir(graphs_path)
        expected_num_graphs = (
            len(binary_cols)
            + len(numerical_cols)
            + len(one_hot_cols)
            + len(frequency_cols)
            + len(ordinal_cols)
            + 2
        )
        assert len(graphs_files) == expected_num_graphs
        assert "binary_feature.png" in graphs_files
        assert "continuous_feature.png" in graphs_files
        assert "one_hot_feature.png" in graphs_files
        assert "frequency_feature.png" in graphs_files
        assert "ordinal_feature.png" in graphs_files
        assert "target.png" in graphs_files
        assert "correlation_matrix.png" in graphs_files

    def test_classical_features_preprocessing(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
            cardinality_threshold=6,
            max_unique_ordinal=8,
        )
        training_preprocessing.data_overview()
        training_preprocessing.drop_duplicates()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        training_preprocessing.drop_col(X_train, X_val, col_drop)
        (
            binary_cols,
            numerical_cols,
            one_hot_cols,
            frequency_cols,
            ordinal_cols,
            _,
        ) = training_preprocessing.detect_column_types(X_train, info)
        X_train_processed, X_val_processed = (
            training_preprocessing.features_preprocessing(
                X_train,
                X_val,
                info,
                binary_cols,
                numerical_cols,
                one_hot_cols,
                frequency_cols,
                ordinal_cols,
            )
        )
        assert X_train_processed.shape[0] == X_train.shape[0]
        assert X_val_processed.shape[0] == X_val.shape[0]
        assert check_path_exists(self.temp_dir, "training_preprocessor.pkl")
        assert check_path_exists(self.temp_dir, "feature_names.pkl")

    def test_classical_numerical_preprocessing_pipeline(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
            df=self.df_classification[["continuous_feature", "target"]].copy(),
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
            cardinality_threshold=6,
            max_unique_ordinal=8,
        )
        training_preprocessing.data_overview()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        training_preprocessing.drop_col(X_train, X_val, col_drop)
        (
            binary_cols,
            numerical_cols,
            one_hot_cols,
            frequency_cols,
            ordinal_cols,
            _,
        ) = training_preprocessing.detect_column_types(X_train, info)
        X_train_processed, X_val_processed = (
            training_preprocessing.features_preprocessing(
                X_train,
                X_val,
                info,
                binary_cols,
                numerical_cols,
                one_hot_cols,
                frequency_cols,
                ordinal_cols,
            )
        )
        imputer = SimpleImputer(strategy="median")
        iqr_trasnform = IQRTransform(info, ["continuous_feature"])
        scaler = StandardScaler()
        X_train_manual = X_train[["continuous_feature"]].values
        X_val_manual = X_val[["continuous_feature"]].values
        X_train_imputed = imputer.fit_transform(X_train_manual)
        X_val_imputed = imputer.transform(X_val_manual)
        iqr_trasnform.fit(X_train_imputed)
        X_train_iqr = iqr_trasnform.transform(X_train_imputed)
        X_val_iqr = iqr_trasnform.transform(X_val_imputed)
        X_train_scaled = scaler.fit_transform(X_train_iqr)
        X_val_scaled = scaler.transform(X_val_iqr)
        assert X_train_processed.shape == X_train_scaled.shape
        assert X_val_processed.shape == X_val_scaled.shape
        assert np.allclose(X_train_processed, X_train_scaled, atol=1e-6)
        assert np.allclose(X_val_processed, X_val_scaled, atol=1e-6)

    def test_classical_binary_preprocessing_pipeline(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
            df=self.df_classification[["binary_feature", "target"]].copy(),
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
            cardinality_threshold=6,
            max_unique_ordinal=8,
        )
        training_preprocessing.data_overview()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        training_preprocessing.drop_col(X_train, X_val, col_drop)
        (
            binary_cols,
            numerical_cols,
            one_hot_cols,
            frequency_cols,
            ordinal_cols,
            _,
        ) = training_preprocessing.detect_column_types(X_train, info)
        X_train_processed, X_val_processed = (
            training_preprocessing.features_preprocessing(
                X_train,
                X_val,
                info,
                binary_cols,
                numerical_cols,
                one_hot_cols,
                frequency_cols,
                ordinal_cols,
            )
        )
        imputer = SimpleImputer(strategy="most_frequent")
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        X_train_manual = X_train[["binary_feature"]].values
        X_val_manual = X_val[["binary_feature"]].values
        X_train_imputed = imputer.fit_transform(X_train_manual)
        X_val_imputed = imputer.transform(X_val_manual)
        X_train_encoded = ordinal_encoder.fit_transform(X_train_imputed)
        X_val_encoded = ordinal_encoder.transform(X_val_imputed)
        assert X_train_processed.shape == X_train_encoded.shape
        assert X_val_processed.shape == X_val_encoded.shape
        assert np.allclose(X_train_processed, X_train_encoded, atol=1e-6)
        assert np.allclose(X_val_processed, X_val_encoded, atol=1e-6)

    def test_classical_one_hot_preprocessing_pipeline(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
            df=self.df_classification[["one_hot_feature", "target"]].copy(),
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
            cardinality_threshold=6,
            max_unique_ordinal=8,
        )
        training_preprocessing.data_overview()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        training_preprocessing.drop_col(X_train, X_val, col_drop)
        (
            binary_cols,
            numerical_cols,
            one_hot_cols,
            frequency_cols,
            ordinal_cols,
            _,
        ) = training_preprocessing.detect_column_types(X_train, info)
        X_train_processed, X_val_processed = (
            training_preprocessing.features_preprocessing(
                X_train,
                X_val,
                info,
                binary_cols,
                numerical_cols,
                one_hot_cols,
                frequency_cols,
                ordinal_cols,
            )
        )
        imputer = SimpleImputer(strategy="most_frequent")
        one_hot_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",
        )
        X_train_manual = X_train[["one_hot_feature"]].values
        X_val_manual = X_val[["one_hot_feature"]].values
        X_train_imputed = imputer.fit_transform(X_train_manual)
        X_val_imputed = imputer.transform(X_val_manual)
        X_train_onehot = one_hot_encoder.fit_transform(X_train_imputed)
        X_val_onehot = one_hot_encoder.transform(X_val_imputed)
        assert X_train_processed.shape == X_train_onehot.shape
        assert X_val_processed.shape == X_val_onehot.shape
        assert np.allclose(X_train_processed, X_train_onehot, atol=1e-6)
        assert np.allclose(X_val_processed, X_val_onehot, atol=1e-6)
        assert X_train_processed.shape[1] == X_train["one_hot_feature"].nunique() - 1
        assert X_val_processed.shape[1] == X_train["one_hot_feature"].nunique() - 1

    def test_classical_frequency_preprocessing_pipeline(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
            df=self.df_classification[["frequency_feature", "target"]].copy(),
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
            cardinality_threshold=6,
            max_unique_ordinal=8,
        )
        training_preprocessing.data_overview()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        training_preprocessing.drop_col(X_train, X_val, col_drop)
        (
            binary_cols,
            numerical_cols,
            one_hot_cols,
            frequency_cols,
            ordinal_cols,
            _,
        ) = training_preprocessing.detect_column_types(X_train, info)
        X_train_processed, X_val_processed = (
            training_preprocessing.features_preprocessing(
                X_train,
                X_val,
                info,
                binary_cols,
                numerical_cols,
                one_hot_cols,
                frequency_cols,
                ordinal_cols,
            )
        )
        imputer = SimpleImputer(strategy="most_frequent")
        frequency_encoder = FrequencyEncoderTransform()
        X_train_manual = X_train[["frequency_feature"]].values
        X_val_manual = X_val[["frequency_feature"]].values
        X_train_imputed = imputer.fit_transform(X_train_manual)
        X_val_imputed = imputer.transform(X_val_manual)
        X_train_processed_manual = frequency_encoder.fit_transform(X_train_imputed)
        X_val_processed_manual = frequency_encoder.transform(X_val_imputed)
        assert np.allclose(
            X_train_processed[:, 0], X_train_processed_manual.ravel(), atol=1e-6
        )
        assert np.allclose(
            X_val_processed[:, 0], X_val_processed_manual.ravel(), atol=1e-6
        )

    def test_classical_target_classification_preprocessing(self):
        training_preprocessing = ClassicalTrainingPreprocessing(
            df=self.df_classification,
            target_col="target",
            problem_type="classification",
            folder_path=self.temp_dir,
            cardinality_threshold=6,
            max_unique_ordinal=8,
        )
        training_preprocessing.data_overview()
        training_preprocessing.drop_duplicates()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        training_preprocessing.drop_col(X_train, X_val, col_drop)
        (
            binary_cols,
            numerical_cols,
            one_hot_cols,
            frequency_cols,
            ordinal_cols,
            _,
        ) = training_preprocessing.detect_column_types(X_train, info)
        X_train_processed, X_val_processed = (
            training_preprocessing.features_preprocessing(
                X_train,
                X_val,
                info,
                binary_cols,
                numerical_cols,
                one_hot_cols,
                frequency_cols,
                ordinal_cols,
            )
        )
        y_train_processed, y_val_processed = (
            training_preprocessing.target_preprocessing(y_train, y_val, info)
        )
        mode = y_train.mode()[0]
        label_encoder = LabelEncoder()
        y_train_filled = y_train.fillna(mode)
        y_val_filled = y_val.fillna(mode)
        y_train_encoded = label_encoder.fit_transform(y_train_filled)
        y_val_encoded = label_encoder.transform(y_val_filled)
        assert y_train_processed.shape == y_train_encoded.shape
        assert y_val_processed.shape == y_val_encoded.shape
        assert np.allclose(y_train_processed, y_train_encoded, atol=1e-6)
        assert np.allclose(y_val_processed, y_val_encoded, atol=1e-6)
        assert check_path_exists(self.temp_dir, "target_preprocessor.pkl")
        assert check_path_exists(self.temp_dir, "target_info.pkl")

    def test_classical_target_regression_preprocessing(self):
        df_regression = self.df_classification.copy()
        df_regression["target"] = df_regression["continuous_feature"] + np.random.randn(
            len(df_regression)
        )
        training_preprocessing = ClassicalTrainingPreprocessing(
            df=df_regression,
            target_col="target",
            problem_type="regression",
            folder_path=self.temp_dir,
            cardinality_threshold=6,
            max_unique_ordinal=8,
        )
        training_preprocessing.data_overview()
        training_preprocessing.drop_duplicates()
        X_train, X_val, y_train, y_val = training_preprocessing.split_data()
        info, col_drop = training_preprocessing.get_data_info(X_train, y_train)
        training_preprocessing.drop_col(X_train, X_val, col_drop)
        (
            binary_cols,
            numerical_cols,
            one_hot_cols,
            frequency_cols,
            ordinal_cols,
            _,
        ) = training_preprocessing.detect_column_types(X_train, info)
        X_train_processed, X_val_processed = (
            training_preprocessing.features_preprocessing(
                X_train,
                X_val,
                info,
                binary_cols,
                numerical_cols,
                one_hot_cols,
                frequency_cols,
                ordinal_cols,
            )
        )
        y_train_processed, y_val_processed = (
            training_preprocessing.target_preprocessing(y_train, y_val, info)
        )
        median = y_train.median()
        y_train_filled = y_train.fillna(median)
        y_val_filled = y_val.fillna(median)
        stander = StandardScaler()
        y_train_scaled = stander.fit_transform(
            y_train_filled.values.reshape(-1, 1)
        ).ravel()
        y_val_scaled = stander.transform(y_val_filled.values.reshape(-1, 1)).ravel()
        assert y_train_processed.shape == y_train_scaled.shape
        assert y_val_processed.shape == y_val_scaled.shape
        assert np.allclose(y_train_processed, y_train_scaled, atol=1e-6)
        assert np.allclose(y_val_processed, y_val_scaled, atol=1e-6)
        assert check_path_exists(self.temp_dir, "target_preprocessor.pkl")
        assert check_path_exists(self.temp_dir, "target_info.pkl")
