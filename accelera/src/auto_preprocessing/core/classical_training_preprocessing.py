import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler

from accelera.src.auto_preprocessing.wrappers.categorical_classification import (
    CategoricalClassification,
)
from accelera.src.auto_preprocessing.wrappers.categorical_regression import (
    CategoricalRegression,
)
from accelera.src.auto_preprocessing.wrappers.correlation_graph import (
    CorrelationGraph,
)
from accelera.src.auto_preprocessing.wrappers.frequency_encoder_transform import (
    FrequencyEncoderTransform,
)
from accelera.src.auto_preprocessing.wrappers.numerical_classification import (
    NumericalClassification,
)
from accelera.src.auto_preprocessing.wrappers.numerical_regression import (
    NumericalRegression,
)
from accelera.src.auto_preprocessing.wrappers.ordinal_classification import (
    OrdinalClassification,
)
from accelera.src.auto_preprocessing.wrappers.ordinal_regression import (
    OrdinalRegression,
)
from accelera.src.auto_preprocessing.wrappers.tabular_preprocessing_report import (
    TabularPreprocessingReport,
)
from accelera.src.auto_preprocessing.wrappers.target_classification import (
    TargetClassification,
)
from accelera.src.auto_preprocessing.wrappers.target_regression import (
    TargetRegression,
)
from accelera.src.utils.preprocessing import drop_columns
from accelera.src.utils.preprocessing import save_pickle

from .training_tabular_preprocessing_base import TrainingTabularPreprocessingBase


class ClassicalTrainingPreprocessing(TrainingTabularPreprocessingBase):
    def __init__(
        self,
        df,
        target_col: str,
        problem_type="classification",
        folder_path=None,
        val_size=0.2,
        random_state=42,
        cardinality_threshold=8,
        max_unique_ordinal=10,
        missing_threshold=0.5,
        columns_need_to_drop=[],
        feature_importance_threshold=0.005,
        is_report=True,
    ):
        super().__init__(
            df,
            target_col,
            val_size,
            random_state,
            folder_path,
            problem_type=problem_type,
        )
        self.problem_type = problem_type
        self.cardinality_threshold = cardinality_threshold
        self.max_unique_ordinal = max_unique_ordinal
        self.missing_threshold = missing_threshold
        self.is_report = is_report
        self.columns_need_to_drop = columns_need_to_drop
        self.feature_importance_threshold = feature_importance_threshold
        if self.problem_type is None:
            raise ValueError("problem_type cannot be None")
        self.problem_type = problem_type.lower()
        if self.problem_type not in ["classification", "regression"]:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'"
            )
        if self.problem_type == "classification" and not (
            np.issubdtype(self.target_type, np.integer)
            or self.target_type == "object"
            or self.target_type == "bool"
        ):
            raise ValueError(
                "Target must be integer or object fro classification problem"
            )
        if not isinstance(self.columns_need_to_drop, list):
            raise ValueError("columns_need_to_drop must be a list")

        if self.problem_type == "regression" and (
            not np.issubdtype(self.target_type, np.number)
            or self.target_type == "bool"
            or self.df[self.target_col].nunique() == 2
        ):
            raise ValueError("Target must be numeric for regression problem")

        if (
            not isinstance(self.cardinality_threshold, int)
            or self.cardinality_threshold < 0
        ):
            raise ValueError("cardinality_threshold must be positive integer")

        if (
            not isinstance(self.max_unique_ordinal, int)
            or self.max_unique_ordinal < 0
        ):
            raise ValueError("max_unique_ordinal must be positive integer")
        if not isinstance(self.missing_threshold, (float, int)) or not (
            0 <= self.missing_threshold <= 1
        ):
            raise ValueError("missing_threshold must be float between 0 and 1")
        if not isinstance(self.feature_importance_threshold, (float, int)) or not (
            0 <= self.feature_importance_threshold <= 1
        ):
            raise ValueError(
                "feature_importance_threshold must be float between 0 and 1"
            )

        save_pickle(self.folder_path, self.df.columns.tolist(), "data_columns.pkl")

    def is_drop_column(self, info, col):
        if col in self.columns_need_to_drop:
            print(col)
            return True, "Column Inside user given list to drop"
        if info[col].get("is_constant", False):
            return True, "The column is constant"

        if info[col].get("p_missing", 0) > self.missing_threshold:
            return (
                True,
                f"Missing above missing_threshold {self.missing_threshold}",
            )
        return False, None

    def outliers_info(self, info, col):
        Q1 = info[col]["Q1"]
        Q3 = info[col]["Q3"]
        min_value = info[col]["min"]
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        lower = max(lower, min_value)
        upper = Q3 + 1.5 * IQR
        return (lower, upper)

    def check_binary(self, col, info):
        if info[col].get("n_unique", 0) == 2 or info[col].get("dtype") == "bool":
            return True

    def get_data_info(self, X_train, y_train):
        col_drop = {}
        info = {}
        df_new = X_train.copy()
        df_new[self.target_col] = y_train
        for col in list(df_new.columns):
            info[col] = {
                "dtype": df_new[col].dtype,
                "n_unique": df_new[col].nunique(),
                "p_unique": df_new[col].nunique() / df_new.shape[0],
                "missing": df_new[col].isna().sum(),
                "p_missing": df_new[col].isna().sum() / df_new.shape[0],
                "is_constant": df_new[col].nunique() == 1,
            }

            if np.issubdtype(df_new[col].dtype, np.number):
                info[col]["Q1"] = df_new[col].quantile(0.25)
                info[col]["Q3"] = df_new[col].quantile(0.75)
                info[col]["min"] = df_new[col].min()
                info[col]["max"] = df_new[col].max()
                info[col]["median"] = df_new[col].median()
                info[col]["mean"] = df_new[col].mean()
                info[col]["outliers_info"] = self.outliers_info(info, col)
            mode = df_new[col].mode()
            if not mode.empty:
                info[col]["mode"] = mode[0]
            else:
                info[col]["mode"] = None

            if col != self.target_col:
                is_drop, reason = self.is_drop_column(info, col)
                if is_drop:
                    col_drop[col] = reason

        return info, col_drop

    def drop_col(self, X_train, X_val, col_drop):
        drop_columns(X_train, col_drop)
        drop_columns(X_val, col_drop)
        self.report_data["drop_columns"] = {
            "col_drop": col_drop,
            "X_trian_head": X_train.head(),
            "X_val_head": X_val.head(),
        }
        save_pickle(self.folder_path, col_drop, "col_drop.pkl")

    def detect_column_types(self, X_train, info):
        binary_cols = []
        numerical_cols = []
        binay_encoding_cols = []
        frequency_cols = []
        ordinal_cols = []
        others = []
        self.report_data["preprocessing"] = []
        for col in X_train.columns:
            if self.check_binary(col, info):
                info[col]["col_type"] = "binary"
                info[col]["preprossing_steps"] = [
                    "Fill missing with most frequent",
                    "Ordinal encoding",
                ]
                binary_cols.append(col)
            elif np.issubdtype(info[col]["dtype"], np.integer):
                if info[col]["n_unique"] <= self.max_unique_ordinal:
                    info[col]["col_type"] = "ordinal"
                    info[col]["preprossing_steps"] = [
                        "Fill missing with most frequent",
                        "Robust scaling",
                    ]
                    ordinal_cols.append(col)
                else:
                    info[col]["col_type"] = "numerical"
                    info[col]["preprossing_steps"] = [
                        "Fill missing with median",
                        "Robust scaling",
                    ]
                    numerical_cols.append(col)

            elif np.issubdtype(info[col]["dtype"], np.floating):
                info[col]["col_type"] = "continuous"
                info[col]["preprossing_steps"] = [
                    "Fill missing with median",
                    "Robust scaling",
                ]
                numerical_cols.append(col)

            elif info[col]["dtype"] == "object":
                if info[col]["n_unique"] <= self.cardinality_threshold:
                    info[col]["col_type"] = "low level cardinality"
                    info[col]["preprossing_steps"] = [
                        "Fill missing with most frequent",
                        "Binary encoding",
                    ]
                    binay_encoding_cols.append(col)
                else:
                    info[col]["col_type"] = "high level cardinality"
                    info[col]["preprossing_steps"] = [
                        "Fill missing with most frequent",
                        "Frequency encoding",
                    ]
                    frequency_cols.append(col)

            else:
                info[col]["col_type"] = "other"
                info[col]["preprossing_steps"] = (
                    f"Drop column because its type {info[col]['dtype']} "
                    f"is not supported"
                )
                others.append(col)
            self.report_data["preprocessing"].append(
                {
                    "col_name": col,
                    "col_type": info[col]["col_type"],
                    "col_preprocessing": info[col]["preprossing_steps"],
                }
            )
        return (
            binary_cols,
            numerical_cols,
            binay_encoding_cols,
            frequency_cols,
            ordinal_cols,
            others,
        )

    def make_graphs(self, X_train, y_train, info):
        if not self.is_report:
            return

        new_df = X_train.copy()
        new_df[self.target_col] = y_train
        self.report_data["graphs"] = {
            "folder_path": self.folder_path,
            "images_name": [],
        }
        for col in X_train.columns:
            if (
                info[col]["col_type"]
                in ["binary", "low level cardinality", "high level cardinality"]
                and self.problem_type == "classification"
            ):
                graph = CategoricalClassification(
                    new_df,
                    col,
                    target_name=self.target_col,
                    folder_path=self.folder_path,
                )
                graph.build_graph()
                self.report_data["graphs"]["images_name"].append(f"{col}")
            if (
                info[col]["col_type"]
                in ["binary", "low level cardinality", "high level cardinality"]
                and self.problem_type == "regression"
            ):
                graph = CategoricalRegression(
                    new_df,
                    col,
                    target_name=self.target_col,
                    folder_path=self.folder_path,
                )
                graph.build_graph()
                self.report_data["graphs"]["images_name"].append(f"{col}")

            if (
                info[col]["col_type"] == "ordinal"
                and self.problem_type == "classification"
            ):
                graph = OrdinalClassification(
                    new_df,
                    col,
                    target_name=self.target_col,
                    folder_path=self.folder_path,
                )
                graph.build_graph()
                self.report_data["graphs"]["images_name"].append(f"{col}")
            if (
                info[col]["col_type"] == "ordinal"
                and self.problem_type == "regression"
            ):
                graph = OrdinalRegression(
                    new_df,
                    col,
                    target_name=self.target_col,
                    folder_path=self.folder_path,
                )
                graph.build_graph()
                self.report_data["graphs"]["images_name"].append(f"{col}")
            if (
                info[col]["col_type"] in ["continuous", "numerical"]
                and self.problem_type == "classification"
            ):
                graph = NumericalClassification(
                    new_df,
                    col,
                    target_name=self.target_col,
                    folder_path=self.folder_path,
                )
                graph.build_graph()
                self.report_data["graphs"]["images_name"].append(f"{col}")
            if (
                info[col]["col_type"] in ["continuous", "numerical"]
                and self.problem_type == "regression"
            ):
                graph = NumericalRegression(
                    new_df,
                    col,
                    target_name=self.target_col,
                    folder_path=self.folder_path,
                )
                graph.build_graph()
                self.report_data["graphs"]["images_name"].append(f"{col}")
        if self.problem_type == "classification":
            target_graph = TargetClassification(
                new_df,
                col_name=self.target_col,
                target_name=self.target_col,
                folder_path=self.folder_path,
            )
            target_graph.build_graph()
            self.report_data["graphs"]["images_name"].append(self.target_col)
        if self.problem_type == "regression":
            target_graph = TargetRegression(
                new_df,
                col_name=self.target_col,
                target_name=None,
                folder_path=self.folder_path,
            )
            target_graph.build_graph()
            self.report_data["graphs"]["images_name"].append(self.target_col)

        correlation_graph = CorrelationGraph(
            new_df,
            col_name=None,
            target_name=None,
            folder_path=self.folder_path,
        )
        correlation_graph.build_graph()
        self.report_data["graphs"]["images_name"].append("correlation_matrix")

    def features_preprocessing(
        self,
        X_train,
        X_val,
        binary_cols,
        numerical_cols,
        binay_encoding_cols,
        frequency_cols,
        ordinal_cols,
    ):
        numerical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )
        binary_encoder_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("binary_encoder", BinaryEncoder()),
            ]
        )
        frequency_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("frequency_encoder", FrequencyEncoderTransform()),
            ]
        )
        binary_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )
        ordinal_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", RobustScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("onehot", binary_encoder_pipeline, binay_encoding_cols),
                ("numerical", numerical_pipeline, numerical_cols),
                ("binary", binary_pipeline, binary_cols),
                ("frequency", frequency_pipeline, frequency_cols),
                ("ordinal", ordinal_pipeline, ordinal_cols),
            ],
            remainder="drop",
            n_jobs=-1,
        )
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        save_pickle(self.folder_path, preprocessor, "training_preprocessor.pkl")
        farures_names = [
            x.split("__")[-1] for x in preprocessor.get_feature_names_out()
        ]
        save_pickle(self.folder_path, farures_names, "feature_names.pkl")
        X_train_processed = pd.DataFrame(X_train_processed, columns=farures_names)
        X_val_processed = pd.DataFrame(X_val_processed, columns=farures_names)
        self.report_data["after_preprocessing"] = {
            "X_train_processed": X_train_processed.head(),
            "X_val_processed": X_val_processed.head(),
        }

        return X_train_processed, X_val_processed

    def target_preprocessing(self, y_train, y_val, info):
        target_dict = {
            "col_name": self.target_col,
            "problem_type": self.problem_type,
        }
        if self.problem_type == "classification":
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_val = label_encoder.transform(y_val)
            save_pickle(self.folder_path, label_encoder, "target_preprocessor.pkl")
            self.report_data["preprocessing"].append(
                {
                    "col_name": self.target_col,
                    "col_type": self.problem_type,
                    "col_preprocessing": [
                        "Label encoding",
                    ],
                }
            )
        elif self.problem_type == "regression":
            robust_scaler = RobustScaler()
            y_train = robust_scaler.fit_transform(
                y_train.values.reshape(-1, 1)
            ).ravel()
            y_val = robust_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
            save_pickle(self.folder_path, robust_scaler, "target_preprocessor.pkl")
            self.report_data["preprocessing"].append(
                {
                    "col_name": self.target_col,
                    "col_type": self.problem_type,
                    "col_preprocessing": [
                        "Robust scaling",
                    ],
                }
            )
        self.report_data["after_preprocessing"]["y_train_processed"] = pd.DataFrame(
            y_train, columns=[self.target_col]
        ).head()
        self.report_data["after_preprocessing"]["y_val_processed"] = pd.DataFrame(
            y_val, columns=[self.target_col]
        ).head()
        save_pickle(self.folder_path, target_dict, "target_info.pkl")
        return y_train, y_val

    def handel_bool_types(self):
        bool_type_col = (
            self.df.drop(columns=[self.target_col])
            .select_dtypes(include=["bool"])
            .columns
        )
        save_pickle(self.folder_path, bool_type_col, "bool_type_col.pkl")
        if len(bool_type_col) == 0:
            return
        self.df[bool_type_col] = self.df[bool_type_col].astype(int)

    def features_importance(self, X_train, y_train, X_val):
        if self.problem_type == "classification":
            score = mutual_info_classif(
                X_train, y_train, random_state=self.random_state
            )
        else:
            score = mutual_info_regression(
                X_train, y_train, random_state=self.random_state
            )
        features_score = pd.DataFrame(
            {"features_name": X_train.columns, "score": score}
        ).sort_values(by="score", ascending=False)
        selected_features_df = features_score[
            features_score["score"] >= self.feature_importance_threshold
        ]
        if len(selected_features_df) == 0:
            raise ValueError(
                "No feature is exist greatre "
                f"than your given threshold {self.feature_importance_threshold}"
            )

        self.report_data["features_selections"] = {
            "features_importance": features_score,
            "selected_features": selected_features_df,
            "feature_importance_threshold": self.feature_importance_threshold,
        }
        X_train_selected = X_train[selected_features_df["features_name"]]
        X_val_selected = X_val[selected_features_df["features_name"]]
        save_pickle(
            self.folder_path,
            selected_features_df["features_name"],
            "selected_features.pkl",
        )
        return X_train_selected.to_numpy(), X_val_selected.to_numpy()

    def common_preprocessing(self):
        self.handel_bool_types()
        self.data_overview()
        self.drop_target_nulls()
        self.drop_duplicates()
        X_train, X_val, y_train, y_val = self.split_data()
        info, col_drop = self.get_data_info(X_train, y_train)
        self.drop_col(X_train, X_val, col_drop)
        (
            binary_cols,
            numerical_cols,
            binay_encoding_cols,
            frequency_cols,
            ordinal_cols,
            _,
        ) = self.detect_column_types(X_train, info)
        self.make_graphs(X_train, y_train, info)
        X_train, X_val = self.features_preprocessing(
            X_train,
            X_val,
            binary_cols,
            numerical_cols,
            binay_encoding_cols,
            frequency_cols,
            ordinal_cols,
        )
        y_train, y_val = self.target_preprocessing(y_train, y_val, info)
        X_train_selected, X_val_selected = self.features_importance(
            X_train, y_train, X_val
        )
        if self.is_report:
            report = TabularPreprocessingReport(self.folder_path, self.report_data)
            report.execute()
        return X_train_selected, y_train, X_val_selected, y_val
