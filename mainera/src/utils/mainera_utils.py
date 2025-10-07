import inspect
import logging
import sys
import os
import sklearn.metrics as metrics

from mainera.src.wrappers.supervised_metric_wrapper import (
    SupervisedMetricWrapper,
)
from mainera.src.wrappers.unsupervised_metric_wrapper import (
    UnSupervisedMetricWrapper,
)

try:
    import graph
except ImportError as e:
    raise ImportError(
        "The 'graph' C++ module could not be imported. "
        "Please ensure it is built and available in your PYTHONPATH."
    ) from e

interactive = True


def print_msg(message, line_break=True, level="info"):
    global interactive
    if interactive:
        if line_break:
            sys.stdout.write(message + "\n")
        else:
            sys.stdout.write(message)
        sys.stdout.flush()
    else:
        if level == "info":
            logging.info(message)
        elif level == "warning":
            logging.warning(message)
        elif level == "error":
            logging.error(message)
        else:
            logging.debug(message)


def get_metric_object(
    metric_name: str,
):
    if metric_name == "":
        return None
    metric_func = getattr(metrics, metric_name, None)
    return metric_func


def get_correct_metric_class(metric_name, metric, y_true=None,tuple_argums=None, **params):
    signature = inspect.signature(metric)
    parameters = list(signature.parameters.keys())
    has_true_labels = any(
        param in parameters for param in ["y_true", "labels_true"]
    )
    has_predictions = any(
        param in parameters
        for param in ["y_pred", "y_score", "y_proba", "labels_pred"]
    )
    supervised = has_true_labels and has_predictions
    unsupervised = "X" in parameters and "labels" in parameters
    if supervised:
        return SupervisedMetricWrapper(metric_name, metric, y_true,tuple_argums, **params)
    elif unsupervised:
        print_msg(
            f"Using unsupervised metric '{metric_name}' "
            "doesn't require y_true. "
            "y_true will be ignored.",
            level="warning",
        )
        return UnSupervisedMetricWrapper(metric_name, metric,tuple_argums, **params)
    else:
        return None

def check_path_exist(path):
    if os.path.exists(path):
        return True
    else:
        return False
    
def create_folder(folder_path):
    if not check_path_exist(folder_path):
        os.makedirs(folder_path, exist_ok=True)

def serialize(pipeline, filepath):
    graph.serialize_graph(pipeline._Pipeline__graph, filepath)
