import inspect
import logging
import sys

import sklearn.metrics as metrics

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


def get_metric_object(metric_name):
    metric_func = getattr(metrics, metric_name, None)
    return metric_func


def metric_validation(metric_func, metric_name):
    signature = inspect.signature(metric_func)
    parameters = list(signature.parameters.keys())
    if ("y_true" not in parameters) and (
        ("y_pred" not in parameters)
        or ("y_score" not in parameters)
        or ("y_prob" not in parameters)
    ):
        raise ValueError(
            f"Metric '{metric_name}' "
            "does not take (y_true, y_pred) or (y_true, y_score) "
            "or (y_true, y_prob) as arguments."
        )
