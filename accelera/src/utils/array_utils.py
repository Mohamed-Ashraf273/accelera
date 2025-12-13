import numpy as np


def convert_to_array(x, dtype=None):
    if x is None:
        return None
    if dtype is not None:
        return np.asarray(x, dtype=dtype)
    return np.asarray(x)


def validate_array_shape(array, expected_shape, names=None):
    if names is None:
        names = ["array", "reference"]

    if array.shape[0] != expected_shape:
        raise ValueError(
            f"{names[0]} and {names[1]} must have the same number of samples. "
            f"Got {names[0]} with {array.shape[0]} samples and "
            f"{names[1]} with {expected_shape} samples."
        )


def get_array_info(x):
    if x is None:
        return {"shape": None, "dtype": None, "ndim": None, "size": None}

    x = convert_to_array(x)
    return {"shape": x.shape, "dtype": x.dtype, "ndim": x.ndim, "size": x.size}
