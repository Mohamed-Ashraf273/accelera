from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from math import inf
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error

# -------------------------------------------------------------------------------
# test on taitanic dataset
df_titanic = pd.read_csv("Testing_dataset/Titanic-Dataset.csv")
# test one  house price data set
# Download latest version
df_price = pd.read_csv("Testing_dataset/Housing.csv")


# ----------------------------------------------------------
# drop column that is has no variance and cloumns like id and column has missing >0.5
def is_drop_column(df, info, col):
    # Drop column if it is constant column if exists
    if info[col].get("is_constant", False):
        print(f"Drop {col} column it is constant")
        return True
    # drop if it is like ID column
    elif (
        info[col].get("p_unique", 0) > 0.70
        and info[col].get("n_unique", 0) > 20
        and info[col].get("type") != "float64"
    ):
        print(f"Drop {col} column it is likely an ID")
        return True
    # drop column has missing >0.5
    elif info[col].get("p_missing", 0) > 0.5:
        print(f"Drop {col} column it is missing")
        return True
    return False


# get the outliers info
def outliers_info(df, info, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (lower, upper)


# cap the outliers
def cap_outliers(df, info, col):
    if "outliers_info" in info[col]:
        lower, upper = info[col]["outliers_info"]
        df[col] = df[col].clip(lower, upper)


# get each column info
def get_data_info(X_train, y_train, target_col):
    col_drop = []
    info = {}
    df_new = X_train.copy()
    df_new[target_col] = y_train
    for col in list(df_new.columns):
        info[col] = {
            "type": df_new[col].dtype,
            "n_unique": df_new[col].nunique(),
            "p_unique": df_new[col].nunique() / df_new.shape[0],
            "p_missing": df_new[col].isna().sum() / df_new.shape[0],
            "is_constant": df_new[col].nunique() == 1,
        }
        if is_drop_column(df_new, info, col) and col != target_col:
            col_drop.append(col)
            info.pop(col)
            continue
        if info[col]["type"] in ["int64", "float64"]:
            info[col]["skew"] = df_new[col].skew()
            info[col]["variance"] = df_new[col].var()
            info[col]["outliers_info"] = outliers_info(df_new, info, col)
            info[col]["min"] = df_new[col].min()
            info[col]["max"] = df_new[col].max()

    return info, col_drop


def drop_duplicates(df):
    # Drop duplicates rows
    df.drop_duplicates(inplace=True)


def drop_columns(X_train, X_test, col_drop):
    # Drop columns
    X_train.drop(columns=col_drop, inplace=True)
    X_test.drop(columns=col_drop, inplace=True)


# handle missing values imputation
def handle_missing(df, info, col):
    if info[col].get("p_missing", 0) != 0:
        col_type = col_type = info[col].get("type")
        if col_type == "object":
            mode = df[col].mode()[0]
            print(f"{col} Fill missing mode value")
            df[col] = df[col].fillna(mode)
        else:
            median = df[col].median()
            print(f"{col} Fill missing median value")
            df[col] = df[col].fillna(median)


def handle_skew(df, info, col, shift_value=0):
    if "skew" in info[col]:
        skew = info[col]["skew"]
        if abs(skew) > 0.75:
            shift_value = 0 if df[col].min() >= 0 else abs(df[col].min()) + 1
            print(f"{col} Apply log1p to reduce skewness")
            df[col] = df[col].apply(
                lambda x: np.log1p(x + shift_value) if x >= 0 else x
            )
            return True
    return False


# handle encoding categorical features
# if it is binary use label encoder
# if it has <11 unique use one hot encoder


def type_encoding(
    info, col, label_encoding_cols, one_hot_encoding_cols, frequency_encoding_cols
):
    col_type = info[col].get("type")
    if col_type == "object":
        n_unique = info[col].get("n_unique", 0)
        if n_unique == 2:
            label_encoding_cols.append(col)
        elif n_unique < 11:
            one_hot_encoding_cols.append(col)
        else:
            frequency_encoding_cols.append(col)


def frequency_encoding(df, col):
    mapping = df[col].value_counts(normalize=True)
    df[col] = df[col].map(mapping)
    print(f"{col} frequency encoding")


def handle_label_encoding_features(X_train, X_test, colums):
    for col in colums:
        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])
        print(f"{col} label encoder")
    return X_train, X_test


def handle_one_hot_encoding_features(X_train, X_test, colums):
    X_train = pd.get_dummies(X_train, columns=colums, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=colums, drop_first=True)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    print(f"{colums} one hot encoder")
    return X_train, X_test


def handle_frequency_encoding_features(X_train, X_test, colums):
    for col in colums:
        frequency_encoding(X_train, col)
        frequency_encoding(X_test, col)
        X_test = X_test.fillna(0)
    return X_train, X_test


def handle_encoding_features(
    X_train, X_test, label_encoding_cols, one_hot_encoding_cols, frequency_encoding_cols
):
    X_train, X_test = handle_label_encoding_features(
        X_train, X_test, label_encoding_cols
    )
    X_train, X_test = handle_one_hot_encoding_features(
        X_train, X_test, one_hot_encoding_cols
    )
    X_train, X_test = handle_frequency_encoding_features(
        X_train, X_test, frequency_encoding_cols
    )
    return X_train, X_test


# features preprocessing
# - cap outliers
# - handle missing
# - handle skewness
# - handle encoding categorical features
def features_preprocessing(X_train, X_test, info):
    label_encoding_cols = []
    one_hot_encoding_cols = []
    frequency_encoding_cols = []
    for col in list(X_train.columns):
        # Missing
        handle_missing(X_train, info, col)
        handle_missing(X_test, info, col)
        # Outliers
        cap_outliers(X_train, info, col)
        cap_outliers(X_test, info, col)
        # Skewness
        col_min = info[col].get("min", 0)
        shift_value = 0 if col_min >= 0 else abs(col_min) + 1
        handle_skew(X_train, info, col, shift_value)
        handle_skew(X_test, info, col, shift_value)
        # Encoding

        type_encoding(
            info,
            col,
            label_encoding_cols,
            one_hot_encoding_cols,
            frequency_encoding_cols,
        )
    X_train, X_test = handle_encoding_features(
        X_train,
        X_test,
        label_encoding_cols,
        one_hot_encoding_cols,
        frequency_encoding_cols,
    )
    return X_train, X_test


# target preprocessing
# object use label encoder


def target_preprocessing(y_train, y_test, info, col):
    col_type = info[col].get("type")
    shift_value = 0
    if col_type == "object":
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)
        print("label encoder")
    else:
        col_min = info[col].get("min", 0)
        shift_value = 0 if col_min >= 0 else abs(col_min) + 1
        y_train_df = y_train.to_frame()
        y_test_df = y_test.to_frame()
        # handle_skew(y_train_df, info, col, shift_value)
        # handle_skew(y_test_df, info, col, shift_value)
        y_train = y_train_df[col]
        y_test = y_test_df[col]
    return y_train, y_test


df_titanic.info()


def split_data(df, target_col):
    X, y = df.drop(columns=[target_col]), df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def common_preprocessing(df, target_col):
    drop_duplicates(df)
    X_train, X_test, y_train, y_test = split_data(df, target_col)
    info, col_drop = get_data_info(X_train, y_train, target_col)
    drop_columns(X_train, X_test, col_drop)
    X_train, X_test = features_preprocessing(X_train, X_test, info)
    y_train, y_test = target_preprocessing(y_train, y_test, info, target_col)
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = common_preprocessing(df_titanic, "Survived")

print(X_train)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Tatianic Score", model.score(X_test, y_test))
# --------------------------------------------------------------
X_train, y_train, X_test, y_test = common_preprocessing(df_price, "price")
#
print("House price data")
print(X_train)
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

model.score(X_test, y_test)
#

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("House price Score", model.score(X_test, y_test))
print("MAE:", mean_absolute_error(y_test, model.predict(X_test)))
#