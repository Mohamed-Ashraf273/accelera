import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ----------------------------------------------------------
# test on taitanic dataset
df_titanic = pd.read_csv("../../../../examples/Titanic-Dataset.csv")
# test one  house price data set
df_price = pd.read_csv("../../../../examples/Housing.csv")
# test on multi-class classification dataset
df_multi_class = pd.read_csv("../../../../examples/train.csv")


# ----------------------------------------------------------
# drop column that is has no
# variance and cloumns like id and column has missing >0.5
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
def outliers_info(info, col):
    Q1 = info[col]["Q1"]
    Q3 = info[col]["Q3"]
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (lower, upper)


# cap the outliers
def cap_outliers(df, info, col):
    if (
        "outliers_info" in info[col]
        and info[col]["type"] in ["float64", "int64"]
        and not info[col].get("binary", False)
    ):
        lower, upper = info[col]["outliers_info"]
        df[col] = df[col].clip(lower, upper)


def check_binary(col, info):
    if info[col].get("n_unique", 0) == 2 or info[col].get("type") == "bool":
        return True


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
            "missing": df_new[col].isna().sum(),
            "p_missing": df_new[col].isna().sum() / df_new.shape[0],
            "is_constant": df_new[col].nunique() == 1,
        }
        print(f"Processing column: {col},{info[col]['n_unique']} unique values")
        if is_drop_column(df_new, info, col) and col != target_col:
            col_drop.append(col)
            info.pop(col)
            continue
        info[col]["binary"] = check_binary(col, info)
        if info[col]["type"] in ["int64", "float64"]:
            info[col]["skew"] = df_new[col].skew()
            info[col]["variance"] = df_new[col].var()
            info[col]["Q1"] = df_new[col].quantile(0.25)
            info[col]["Q3"] = df_new[col].quantile(0.75)
            info[col]["outliers_info"] = outliers_info(info, col)
            info[col]["min"] = df_new[col].min()
            info[col]["max"] = df_new[col].max()
            info[col]["median"] = df_new[col].median()
            info[col]["mean"] = df_new[col].mean()
        if info[col]["type"] == "object" or info[col]["binary"]:
            info[col]["distribution"] = (
                df_new[col].value_counts(normalize=True).to_dict()
            )
            info[col]["mode"] = df_new[col].mode()[0]

    return info, col_drop


def drop_duplicates(df):
    # Drop duplicates rows
    df.drop_duplicates(inplace=True)


def drop_columns(X_train, X_test, col_drop):
    # Drop columns
    X_train.drop(columns=col_drop, inplace=True)
    X_test.drop(columns=col_drop, inplace=True, errors="ignore")


# handle missing values imputation
def handle_missing(df, info, col):
    if info[col].get("p_missing", 0) != 0:
        col_type = col_type = info[col].get("type")
        if col_type in ["object"] or info[col].get("binary", False):
            mode = info[col]["mode"]
            print(f"{col} Fill missing mode value")
            df[col] = df[col].fillna(mode)
        else:
            median = info[col]["median"]
            print(f"{col} Fill missing median value")
            df[col] = df[col].fillna(median)


# handle encoding categorical features
# if it is binary use label encoder
# if it has <11 unique use one hot encoder


def type_encoding(
    info,
    col,
    label_encoding_cols,
    one_hot_encoding_cols,
    frequency_encoding_cols,
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


def frequency_encoding(df, col, mapping):
    df[col] = df[col].map(mapping)
    df[col] = df[col].fillna(0)
    print(f"{col} frequency encoding")


def handle_label_encoding_features(X_train, X_test, colums):
    for col in colums:
        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])
        print(f"{col} label encoder")
    return X_train, X_test


def handle_one_hot_encoding_features(X_train, X_test, columns):
    X_train_dummies = pd.get_dummies(
        X_train[columns], drop_first=True, dtype=int
    )
    X_test_dummies = pd.get_dummies(X_test[columns], drop_first=True, dtype=int)
    X_test_dummies = X_test_dummies.reindex(
        columns=X_train_dummies.columns, fill_value=0
    )
    X_train = X_train.drop(columns=columns).join(X_train_dummies)
    X_test = X_test.drop(columns=columns).join(X_test_dummies)
    print(f"One-hot encoded {columns}")
    return X_train, X_test


def handle_frequency_encoding_features(X_train, X_test, info, colums):
    for col in colums:
        mapping = info[col]["distribution"]
        frequency_encoding(X_train, col, mapping)
        frequency_encoding(X_test, col, mapping)

    return X_train, X_test


def handle_encoding_features(
    X_train,
    X_test,
    info,
    label_encoding_cols,
    one_hot_encoding_cols,
    frequency_encoding_cols,
):
    X_train, X_test = handle_label_encoding_features(
        X_train, X_test, label_encoding_cols
    )
    X_train, X_test = handle_one_hot_encoding_features(
        X_train, X_test, one_hot_encoding_cols
    )
    X_train, X_test = handle_frequency_encoding_features(
        X_train, X_test, info, frequency_encoding_cols
    )
    return X_train, X_test


# features preprocessing
# - cap outliers
# - handle missing
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
        info,
        label_encoding_cols,
        one_hot_encoding_cols,
        frequency_encoding_cols,
    )
    return X_train, X_test


# target preprocessing
# object use label encoder


def target_preprocessing(y_train, y_test, info, col):
    col_type = info[col].get("type")
    if col_type == "object":
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)
        print("label encoder")
    else:
        y_train_df = y_train.to_frame()
        y_test_df = y_test.to_frame()
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


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Tatianic Score", model.score(X_test, y_test))
# --------------------------------------------------------------
X_train, y_train, X_test, y_test = common_preprocessing(df_price, "price")

print("House price data")
print(X_train)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

model.score(X_test, y_test)


model = LinearRegression()
model.fit(X_train, y_train)

print("House price Score", model.score(X_test, y_test))
print("MAE:", mean_absolute_error(y_test, model.predict(X_test)))
print("--------------------------------------------------")
print(df_multi_class.head())
X_train, y_train, X_test, y_test = common_preprocessing(
    df_multi_class, "Segmentation"
)
print("Multi-class data")
print(X_train)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Multi-class Score", model.score(X_test, y_test))
xgb_model = XGBClassifier(eval_metric="mlogloss", random_state=42)

xgb_model.fit(X_train, y_train)
print("Multi-class XGB Score", xgb_model.score(X_test, y_test))
