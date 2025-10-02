import inspect
import logging
import sys

import sklearn.metrics as metrics

from mainera.src.wrappers.metric_wrapper import SupervisedMetricWrapper
from mainera.src.wrappers.metric_wrapper import UnSupervisedMetricWrapper

interactive = True


def print_msg(message, line_break=True, level="info"):
    """Print an mAInera message to stdout or logging."""
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


def get_correct_metric_class(metric_name, metric, y_true=None, X=None, **params):
    signature = inspect.signature(metric)
    parameters = list(signature.parameters.keys())
    print(parameters)
    has_true_labels = any(param in parameters for param in ["y_true", "labels_true"])
    has_predictions = any(
        param in parameters for param in ["y_pred", "y_score", "y_proba", "labels_pred"]
    )
    supervised = has_true_labels and has_predictions
    unsupervised = "X" in parameters and "labels" in parameters
    if supervised:
        return SupervisedMetricWrapper(metric_name, metric, y_true, **params)
    elif unsupervised:
        return UnSupervisedMetricWrapper(metric_name, metric, X, **params)
    else:
        return None
