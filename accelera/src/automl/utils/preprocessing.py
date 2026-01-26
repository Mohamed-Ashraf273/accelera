from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from accelera.src.automl.wrappers.IQR_transform import IQRTransform
from accelera.src.automl.wrappers.frequency_encoder_transform import (
    FrequencyEncoderTransform,
)
import numpy as np


def is_drop_column(info, col):
    # drop column if it is constent, percent of unique values > 90% (likely ID),
    # or percent of missing > 50%
    if info[col].get("is_constant", False):
        print(f"Drop {col} column it is constant")
        return True
    elif info[col].get("p_unique", 0) > 0.90 and info[col].get("dtype") != "float64":
        print(f"Drop {col} column it is likely an ID")
        return True
    elif info[col].get("p_missing", 0) > 0.5:
        print(f"Drop {col} column it is missing")
        return True
    return False


def outliers_info(info, col):
    # get lower and upper bounds of outliers
    Q1 = info[col]["Q1"]
    Q3 = info[col]["Q3"]
    min_value = info[col]["min"]
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    lower = max(lower, min_value)
    upper = Q3 + 1.5 * IQR
    return (lower, upper)


def check_binary(col, info):
    if info[col].get("n_unique", 0) == 2 or info[col].get("dtype") == "bool":
        return True


def get_data_info(X_train, y_train, target_col):
    # this function return info about the training data like mean, median, dtype, n_unique, p_unique, missing, p_missing
    col_drop = []
    info = {}
    df_new = X_train.copy()
    df_new[target_col] = y_train
    for col in list(df_new.columns):
        info[col] = {
            "dtype": df_new[col].dtype,
            "n_unique": df_new[col].nunique(),
            "p_unique": df_new[col].nunique() / df_new.shape[0],
            "missing": df_new[col].isna().sum(),
            "p_missing": df_new[col].isna().sum() / df_new.shape[0],
            "is_constant": df_new[col].nunique() == 1,
        }
        if is_drop_column(info, col) and col != target_col:
            col_drop.append(col)
            info.pop(col)
            continue
        if info[col]["dtype"] in ["int64", "float64"]:
            info[col]["skew"] = df_new[col].skew()
            info[col]["variance"] = df_new[col].var()
            info[col]["Q1"] = df_new[col].quantile(0.25)
            info[col]["Q3"] = df_new[col].quantile(0.75)
            info[col]["min"] = df_new[col].min()
            info[col]["max"] = df_new[col].max()
            info[col]["median"] = df_new[col].median()
            info[col]["mean"] = df_new[col].mean()
            info[col]["outliers_info"] = outliers_info(info, col)

        mode = df_new[col].mode()
        if not mode.empty:
            info[col]["mode"] = mode[0]
        else:
            info[col]["mode"] = None

    return info, col_drop


def detect_column_types(
    X_train,
    info,
    text_threshold=40,
    one_hot_threshold=10,
    max_unique_ordinal=7,
    categorical_ratio_threshold=0.05,
):
    binary_cols = []
    numerical_cols = []
    one_hot_cols = []
    frequency_cols = []
    text_cols = []
    ordinal_cols = []
    others = []
    for col in X_train.columns:
        # first check if binary
        is_binary = check_binary(col, info)
        if is_binary:
            info[col]["col_type"] = "binary"
            binary_cols.append(col)
            continue

        # if it is integer it may be ordinal, categorical_frequency or numerical
        if np.issubdtype(info[col]["dtype"], np.integer):
            if info[col]["n_unique"] <= max_unique_ordinal:
                sorted_unique_values = np.sort(X_train[col].dropna().unique())
                diff = np.diff(sorted_unique_values)
                if np.all(diff == 1):
                    info[col]["col_type"] = "ordinal"
                    ordinal_cols.append(col)
                else:
                    info[col]["col_type"] = "categorical_frequency"
                    frequency_cols.append(col)

            elif info[col]["p_unique"] < categorical_ratio_threshold:
                info[col]["col_type"] = "categorical_frequency"
                frequency_cols.append(col)

            else:
                info[col]["col_type"] = "numerical"
                numerical_cols.append(col)
        # if it is float it is continuous

        elif np.issubdtype(info[col]["dtype"], np.floating):
            info[col]["col_type"] = "continuous"
            numerical_cols.append(col)
        # if it is object may be categorical(one hot, frequency) or text

        elif info[col]["dtype"] == "object":
            n_unique = info[col]["n_unique"]
            if n_unique <= one_hot_threshold:
                info[col]["col_type"] = "categorical_one_hot"
                one_hot_cols.append(col)
            else:
                avg_length = X_train[col].dropna().apply(lambda x: len(str(x))).mean()
                if avg_length > text_threshold:
                    info[col]["col_type"] = "text"
                    text_cols.append(col)
                else:
                    info[col]["col_type"] = "categorical_frequency"
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


def drop_duplicates(df):
    df.drop_duplicates(inplace=True)


def lower_data(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.lower()


def drop_columns(X_train, X_val, col_drop):
    X_train.drop(columns=col_drop, inplace=True)
    X_val.drop(columns=col_drop, inplace=True, errors="ignore")


def features_preprocessing(X_train, X_val, info):
    (
        binary_cols,
        numerical_cols,
        one_hot_cols,
        frequency_cols,
        text_cols,
        ordinal_cols,
        others,
    ) = detect_column_types(X_train, info)
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
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
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
    text_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "extract_column",
                FunctionTransformer(
                    lambda x: np.array([" ".join(row.astype(str)) for row in x]),
                    validate=False,
                ),
            ),
            ("tfidf_vectorizer", TfidfVectorizer(max_features=1000)),
        ]
    )
    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder()),
        ]
    )
    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_pipeline, text_cols),
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
    return X_train_processed, X_val_processed


def target_preprocessing(y_train, y_val, info, col, problem_type):
    if problem_type == "classification":
        y_train = y_train.fillna(info[col]["mode"])
        y_val = y_val.fillna(info[col]["mode"])
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_val = label_encoder.transform(y_val)
        return y_train, y_val, label_encoder
    elif problem_type == "regression":
        y_train = y_train.fillna(info[col]["median"])
        y_val = y_val.fillna(info[col]["median"])
        stander_scaler = StandardScaler()
        y_train = stander_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val = stander_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
        return y_train, y_val, stander_scaler


def split_data(df, target_col):
    X, y = df.drop(columns=[target_col]), df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val


def common_preprocessing(df, target_col: str, problem_type="classification"):
    # 1- drop duplictes
    # 2- lower data
    # 3- split data
    # 4- get data info
    # 5- drop columns
    # 6- features preprocessing
    # 7- target preprocessing
    if problem_type not in ["classification", "regression"]:
        raise ValueError("problem_type must be either 'classification' or 'regression'")
    drop_duplicates(df)
    lower_data(df)
    X_train, X_val, y_train, y_val = split_data(df, target_col)
    info, col_drop = get_data_info(X_train, y_train, target_col)
    drop_columns(X_train, X_val, col_drop)
    X_train, X_val = features_preprocessing(X_train, X_val, info)
    y_train, y_val, _ = target_preprocessing(
        y_train, y_val, info, target_col, problem_type
    )
    return X_train, y_train, X_val, y_val
