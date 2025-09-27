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


def get_metric_object(metric_name: str):
    if metric_name == "":
        return None
    metric_func = getattr(metrics, metric_name, None)
    return metric_func


def metric_validation(metric_func, metric_name):
    signature = inspect.signature(metric_func)
    parameters = list(signature.parameters.keys())

    # Check if it has y_true AND at least one of y_pred, y_score, or y_prob
    has_y_true = "y_true" in parameters
    has_pred_or_score = any(
        param in parameters for param in ["y_pred", "y_score", "y_prob"]
    )

    if not (has_y_true and has_pred_or_score):
        raise ValueError(
            f"Metric '{metric_name}' "
            "does not take (y_true, y_pred) or (y_true, y_score) "
            "or (y_true, y_prob) as arguments."
        )


def get_learning_type(model) -> str:
    if not hasattr(model, "fit") or not callable(getattr(model, "fit")):
        return "unknown"

    try:
        sig = inspect.signature(model.fit)

        has_required_y = any(
            param_name == "y" and param.default is param.empty
            for param_name, param in sig.parameters.items()
        )

        if has_required_y:
            return "supervised"
        else:
            return "unsupervised"

    except (ValueError, TypeError):
        return "unknown"
