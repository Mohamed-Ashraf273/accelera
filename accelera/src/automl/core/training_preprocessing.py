from accelera.src.automl.core.preprocessing_base import PreprocessingBase
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from accelera.src.automl.wrappers.IQR_transform import IQRTransform
from accelera.src.automl.wrappers.categorical_regression import CategoricalRegression
from accelera.src.automl.wrappers.frequency_encoder_transform import (
    FrequencyEncoderTransform,
)
from accelera.src.automl.wrappers.categorical_classification import (
    CategoricalClassification,
)
import numpy as np
import os

from accelera.src.automl.wrappers.numerical_classification import (
    NumericalClassification,
)
from accelera.src.automl.wrappers.numerical_regression import NumericalRegression
from accelera.src.automl.wrappers.ordinal_classification import OrdinalClassification
from accelera.src.automl.wrappers.ordinal_regression import OrdinalRegression
from accelera.src.automl.wrappers.preprocessing_report import PreprocessingReport
from accelera.src.automl.wrappers.target_regression import TargetRegression
from accelera.src.automl.wrappers.target_classification import TargetClassification
from accelera.src.automl.wrappers.text_graph import TextGraph
from accelera.src.automl.wrappers.correlation_graph import CorrelationGraph


class TrainingPreprocessing(PreprocessingBase):
    def __init__(
        self,
        df,
        target_col: str,
        problem_type="classification",
        folder_path=None,
        test_size=0.2,
        random_state=42,
        text_threshold=40,
        one_hot_threshold=8,
        max_unique_ordinal=6,
        categorical_ratio_threshold=0.05,
        missing_threshold=0.5,
        unique_threshold=0.9,
    ):
        super().__init__(df, folder_path)
        self.target_col = target_col
        self.problem_type = problem_type
        self.test_size = test_size
        self.random_state = random_state
        self.text_threshold = text_threshold
        self.one_hot_threshold = one_hot_threshold
        self.max_unique_ordinal = max_unique_ordinal
        self.categorical_ratio_threshold = categorical_ratio_threshold
        self.missing_threshold = missing_threshold
        self.unique_threshold = unique_threshold

        if self.problem_type not in ["classification", "regression"]:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'"
            )
        if target_col not in df.columns:
            raise ValueError("target_col must be one of the dataframe columns")
        os.makedirs(self.folder_path, exist_ok=True)
        self.save_pikle(self.df.columns.tolist(), "data_columns.pkl")

    def is_drop_column(self, info, col):
        # drop column if it is constent, percent of unique values > 90% (likely ID),
        # or percent of missing > 50%
        if info[col].get("is_constant", False):
            print(f"Drop {col} column it is constant")
            return True, "The column is constant"
        elif (
            info[col].get("p_unique", 0) > self.unique_threshold
            and info[col].get("dtype") != "float64"
        ):
            print(f"Drop {col} column it is likely an ID")
            return True, f"It is above unique_threshold {self.unique_threshold}"
        elif info[col].get("p_missing", 0) > self.missing_threshold:
            print(f"Drop {col} column it is missing")
            return True, f"missing above missing_threshold {self.missing_threshold}"
        return False, None

    def outliers_info(self, info, col):
        # get lower and upper bounds of outliers
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
        # this function return info about the training data like mean, median, dtype, n_unique, p_unique, missing, p_missing
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

            if info[col]["dtype"] in ["int64", "float64"]:
                info[col]["skew"] = df_new[col].skew()
                info[col]["variance"] = df_new[col].var()
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

    def detect_column_types(self, X_train, info):
        binary_cols = []
        numerical_cols = []
        one_hot_cols = []
        frequency_cols = []
        text_cols = []
        ordinal_cols = []
        others = []
        for col in X_train.columns:
            # first check if binary
            is_binary = self.check_binary(col, info)
            if is_binary:
                info[col]["col_type"] = "binary"
                binary_cols.append(col)
                continue

            # if it is integer it may be ordinal, categorical_frequency or numerical
            if np.issubdtype(info[col]["dtype"], np.integer):
                if info[col]["n_unique"] <= self.max_unique_ordinal:
                    sorted_unique_values = np.sort(X_train[col].dropna().unique())
                    diff = np.diff(sorted_unique_values)
                    if np.all(diff == 1):
                        info[col]["col_type"] = "ordinal"
                        info[col]["preprossing_steps"] = [
                            "fill missing with most frequent",
                            "ordinal_encoding",
                        ]
                        ordinal_cols.append(col)
                    else:
                        info[col]["col_type"] = "categorical_frequency"
                        info[col]["preprossing_steps"] = [
                            "fill missing with most frequent",
                            "frequency_encoding",
                        ]
                        frequency_cols.append(col)

                elif info[col]["p_unique"] < self.categorical_ratio_threshold:
                    info[col]["col_type"] = "categorical_frequency"
                    info[col]["preprossing_steps"] = [
                        "fill missing with most frequent",
                        "frequency_encoding",
                    ]
                    frequency_cols.append(col)

                else:
                    info[col]["col_type"] = "numerical"
                    info[col]["preprossing_steps"] = [
                        "fill missing with median",
                        "IQR_transform",
                        "standered scaling",
                    ]
                    numerical_cols.append(col)
            # if it is float it is continuous

            elif np.issubdtype(info[col]["dtype"], np.floating):
                info[col]["col_type"] = "continuous"
                info[col]["preprossing_steps"] = [
                    "fill missing with median",
                    "IQR_transform",
                    "standered scaling",
                ]
                numerical_cols.append(col)
            # if it is object may be categorical(one hot, frequency) or text

            elif info[col]["dtype"] == "object":
                n_unique = info[col]["n_unique"]
                if n_unique <= self.one_hot_threshold:
                    info[col]["col_type"] = "categorical_one_hot"
                    info[col]["preprossing_steps"] = [
                        "fill missing with most frequent",
                        "one hot encoding",
                    ]
                    one_hot_cols.append(col)
                else:
                    avg_length = (
                        X_train[col].dropna().apply(lambda x: len(str(x))).mean()
                    )
                    if avg_length > self.text_threshold:
                        info[col]["col_type"] = "text"
                        info[col]["preprossing_steps"] = [
                            "fill missing with empty string",
                            "tfidf vectorization",
                        ]
                        text_cols.append(col)
                    else:
                        info[col]["col_type"] = "categorical_frequency"
                        info[col]["preprossing_steps"] = [
                            "fill missing with most frequent",
                            "frequency_encoding",
                        ]
                        frequency_cols.append(col)
            else:
                info[col]["col_type"] = "other"
                others.append(col)
        print("Detected column types:")
        print(f"Binary columns: {binary_cols}")
        print(f"Numerical columns: {numerical_cols}")
        print(f"One-hot columns: {one_hot_cols}")
        print(f"Frequency-encoded columns: {frequency_cols}")
        print(f"Text columns: {text_cols}")
        print(f"Ordinal columns: {ordinal_cols}")
        print(f"Other columns: {others}")
        return (
            binary_cols,
            numerical_cols,
            one_hot_cols,
            frequency_cols,
            text_cols,
            ordinal_cols,
            others,
        )

    def make_graphs(self, X_train, y_train, info):
        new_df = X_train.copy()
        new_df[self.target_col] = y_train

        for col in X_train.columns:
            if (
                info[col]["col_type"]
                in ["binary", "categorical_one_hot", "categorical_frequency"]
                and self.problem_type == "classification"
            ):
                graph = CategoricalClassification(
                    new_df,
                    col,
                    target_name=self.target_col,
                    folder_path=self.folder_path,
                )
                graph.build_graph()
            if (
                info[col]["col_type"]
                in ["binary", "categorical_one_hot", "categorical_frequency"]
                and self.problem_type == "regression"
            ):
                graph = CategoricalRegression(
                    new_df,
                    col,
                    target_name=self.target_col,
                    folder_path=self.folder_path,
                )
                graph.build_graph()
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
            if info[col]["col_type"] == "ordinal" and self.problem_type == "regression":
                graph = OrdinalRegression(
                    new_df,
                    col,
                    target_name=self.target_col,
                    folder_path=self.folder_path,
                )
                graph.build_graph()
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
            if info[col]["col_type"] == "text":
                text_graph = TextGraph(
                    new_df,
                    col_name=col,
                    target_name=self.target_col,
                    folder_path=self.folder_path,
                )
                text_graph.build_graph()
        if self.problem_type == "classification":
            target_graph = TargetClassification(
                new_df,
                col_name=self.target_col,
                target_name=self.target_col,
                folder_path=self.folder_path,
            )
            target_graph.build_graph()
        if self.problem_type == "regression":
            target_graph = TargetRegression(
                new_df,
                col_name=self.target_col,
                target_name=None,
                folder_path=self.folder_path,
            )
            target_graph.build_graph()

        correlation_graph = CorrelationGraph(
            new_df,
            col_name=None,
            target_name=None,
            folder_path=self.folder_path,
        )
        correlation_graph.build_graph()

    def drop_duplicates(self):
        self.df.drop_duplicates(inplace=True)

    def make_text_transformers(self, text_cols):
        transformers = []
        for col in text_cols:
            transform = (
                f"{col}_tfidf",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        (
                            "tfidf",
                            TfidfVectorizer(max_features=1000, stop_words="english"),
                        ),
                    ]
                ),
                [col],
            )
            transformers.append(transform)
        return transformers

    def features_preprocessing(
        self,
        X_train,
        X_val,
        info,
        binary_cols,
        numerical_cols,
        one_hot_cols,
        frequency_cols,
        text_cols,
        ordinal_cols,
    ):

        # Pipelines
        numerical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("iqr_transformer", IQRTransform(info, numerical_cols)),
                ("scaler", StandardScaler()),
            ]
        )
        one_hot_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "one_hot_encoder",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False, drop="first"
                    ),
                ),
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
                ("ordinal", OrdinalEncoder()),
            ]
        )
        text_pipeline = self.make_text_transformers(text_cols)
        ordinal_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder()),
            ]
        )
        # Column Transformer
        preprocessor = ColumnTransformer(
            transformers=[
                *text_pipeline,
                ("onehot", one_hot_pipeline, one_hot_cols),
                ("numerical", numerical_pipeline, numerical_cols),
                ("binary", binary_pipeline, binary_cols),
                ("frequency", frequency_pipeline, frequency_cols),
                ("ordinal", ordinal_pipeline, ordinal_cols),
            ],
            remainder="passthrough",
        )
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        # save the processor
        self.save_pikle(preprocessor, "training_preprocessor.pkl")
        return X_train_processed, X_val_processed

    def target_preprocessing(self, y_train, y_val, info):
        target_dict = {
            "col_name": self.target_col,
            "problem_type": self.problem_type,
        }
        if self.problem_type == "classification":
            y_train = y_train.fillna(info[self.target_col]["mode"])
            y_val = y_val.fillna(info[self.target_col]["mode"])
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_val = label_encoder.transform(y_val)
            self.save_pikle(label_encoder, "target_preprocessor.pkl")
            target_dict["mode"] = info[self.target_col]["mode"]

        elif self.problem_type == "regression":
            y_train = y_train.fillna(info[self.target_col]["median"])
            y_val = y_val.fillna(info[self.target_col]["median"])
            target_dict["median"] = info[self.target_col]["median"]
            stander_scaler = StandardScaler()
            y_train = stander_scaler.fit_transform(
                y_train.values.reshape(-1, 1)
            ).ravel()
            y_val = stander_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
            self.save_pikle(stander_scaler, "target_preprocessor.pkl")
        self.save_pikle(target_dict, "target_info.pkl")
        return y_train, y_val

    def split_data(self):
        X, y = self.df.drop(columns=[self.target_col]), self.df[self.target_col]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_val, y_train, y_val

    def common_preprocessing(self):
        # 1- drop duplictes
        # 2- lower data
        # 3- split data
        # 4- get data info
        # 5- drop columns
        # 6- features preprocessing
        # 7- target preprocessing
        report = PreprocessingReport(self.folder_path, self.df)
        report.execute()
        self.drop_duplicates()
        self.lower_data()
        X_train, X_val, y_train, y_val = self.split_data()
        info, col_drop = self.get_data_info(X_train, y_train)
        # save column drop for testing
        self.drop_columns(X_train, col_drop)

        self.save_pikle(col_drop, "col_drop.pkl")
        (
            binary_cols,
            numerical_cols,
            one_hot_cols,
            frequency_cols,
            text_cols,
            ordinal_cols,
            _,
        ) = self.detect_column_types(X_train, info)

        self.make_graphs(X_train, y_train, info)

        X_train, X_val = self.features_preprocessing(
            X_train,
            X_val,
            info,
            binary_cols,
            numerical_cols,
            one_hot_cols,
            frequency_cols,
            text_cols,
            ordinal_cols,
        )
        y_train, y_val = self.target_preprocessing(y_train, y_val, info)
        return X_train, y_train, X_val, y_val
