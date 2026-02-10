import os
import pickle


def check_path_exists(folder_path, filename):
    path = os.path.join(folder_path, filename)
    if not os.path.exists(path):
        raise ValueError(f"{path} does not exist")
    return True


def save_pickle(folder_path, obj, filename):
    filepath = os.path.join(folder_path, filename)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(folder_path, filename):
    filepath = os.path.join(folder_path, filename)
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


def lower_data(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
            
def drop_columns(X, col_drop):
    col_drop = list(col_drop.keys())
    if col_drop:
        X.drop(columns=col_drop, inplace=True, errors="ignore")
