import pickle

import numpy as np

from accelera.src.utils.source_backed_function import SourceBackedFunction


def nested_shift_for_source_backed_function(X):
    def shift(value):
        return value + 1.0

    return shift(X)


def multiply_for_source_backed_function(value, factor):
    return value * factor


def nested_external_helper_for_source_backed_function(X):
    def scale(value):
        return multiply_for_source_backed_function(value, 2.0)

    return scale(X)


def test_source_backed_function_pickles_nested_helpers():
    source_func = SourceBackedFunction(nested_shift_for_source_backed_function)
    restored_func = pickle.loads(pickle.dumps(source_func))

    X = np.array([1.0, 2.0])

    assert np.array_equal(restored_func(X), np.array([2.0, 3.0]))
    assert source_func.compilation_source().index(
        "def shift"
    ) < source_func.compilation_source().index(
        "def nested_shift_for_source_backed_function"
    )


def test_source_backed_function_bundles_referenced_helpers():
    source_func = SourceBackedFunction(
        nested_external_helper_for_source_backed_function
    )
    restored_func = pickle.loads(pickle.dumps(source_func))
    compilation_source = source_func.compilation_source()

    X = np.array([1.0, 2.0])

    assert np.array_equal(restored_func(X), np.array([2.0, 4.0]))
    assert compilation_source.index(
        "def multiply_for_source_backed_function"
    ) < compilation_source.index("def scale")
    assert compilation_source.index("def scale") < compilation_source.index(
        "def nested_external_helper_for_source_backed_function"
    )
