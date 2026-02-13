import os
import pickle
from PIL import Image
import torch

def check_path_exists(folder_path, filename):
    path = os.path.join(folder_path, filename)
    if not os.path.exists(path):
        raise ValueError(f"{path} does not exist")
    return True


def get_sub_folders_names(folder_path):
    sub_folders_name = [
        class_name
        for class_name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, class_name))
    ]
    return sub_folders_name


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


def is_valid_image(image_path):
    valid_extension = (".jpg", ".png", ".jpeg")

    if not os.path.isfile(image_path):
        return False
    if image_path.endswith(valid_extension):
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    return False
def collect_function(batch):
    images=[item[0]for item in batch]
    labels=[item[1]for item in batch]
    images_stack=torch.stack(images,dim=0)
    if labels[0] is None:
        return images_stack
    labels=torch.tensor(labels,dtype=torch.long)
    return images_stack,labels