import inspect
import logging
import os
import shutil
import sys

import sklearn.metrics as metrics

from accelera.src.wrappers.supervised_metric import SupervisedMetric
from accelera.src.wrappers.unsupervised_metric import UnSupervisedMetric

try:
    from graph import serialize_graph
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


def get_correct_metric_class(
    metric_name,
    metric,
    y_true=None,
    plot_func=None,
    labels_name=None,
    headers_name=None,
    **params,
):
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
        return SupervisedMetric(
            metric_name,
            metric,
            y_true,
            plot_func,
            labels_name,
            headers_name,
            **params,
        )
    elif unsupervised:
        print_msg(
            f"Using unsupervised metric '{metric_name}' "
            "doesn't require y_true. "
            "y_true will be ignored.",
            level="warning",
        )
        return UnSupervisedMetric(
            metric_name, metric, plot_func, labels_name, headers_name, **params
        )
    else:
        return None


def check_path_exist(path):
    if os.path.exists(path):
        return True
    else:
        return False


def create_folder(folder_path):
    if check_path_exist(folder_path):
        shutil.rmtree(folder_path)

    os.makedirs(folder_path, exist_ok=True)


def get_instance_type(instance) -> str:
    def check_method_patterns():
        has_predict = hasattr(instance, "predict")
        has_transform = hasattr(instance, "transform")
        has_fit_predict = hasattr(instance, "fit_predict")
        has_predict_proba = hasattr(instance, "predict_proba")
        has_decision_function = hasattr(instance, "decision_function")
        has_fit = hasattr(instance, "fit")

        supervised_score = 0
        if has_predict_proba:
            supervised_score += 3
        if has_decision_function:
            supervised_score += 2
        if has_predict and not has_transform:
            supervised_score += 2

        unsupervised_score = 0
        if has_fit_predict:
            unsupervised_score += 2
        if has_transform and not has_predict_proba:
            unsupervised_score += 2
        if hasattr(instance, "cluster_centers_"):
            unsupervised_score += 3
        if hasattr(instance, "components_"):
            unsupervised_score += 2

        # Check for preprocessing patterns
        if (
            has_transform
            and has_fit
            and not has_predict
            and not has_fit_predict
        ):
            # This is likely a preprocessor/transformer
            # Check if it needs labels (supervised preprocessing)
            if hasattr(instance, "score_func") or hasattr(instance, "scores_"):
                supervised_score += 2
            elif hasattr(instance, "k") or hasattr(instance, "percentile"):
                supervised_score += 1
            else:
                unsupervised_score += 1

        if supervised_score > unsupervised_score:
            return "supervised"
        elif unsupervised_score > supervised_score:
            return "unsupervised"
        return None

    def check_class_name():
        class_name = instance.__class__.__name__.lower()

        supervised_patterns = [
            "classifier",
            "regressor",
            "svm",
            "svc",
            "svr",
            "logistic",
            "linear",
            "ridge",
            "lasso",
            "tree",
            "forest",
            "boost",
            "bagging",
            "voting",
            "neural",
            # Feature selection (supervised preprocessing)
            "selectkbest",
            "selectpercentile",
            "selectfpr",
            "selectfdr",
            "selectfwe",
            "rfe",
            "rfecv",
            "variancethreshold",
            # Target-based preprocessing
            "targetencoder",
            "ordinalencoder",
        ]

        unsupervised_patterns = [
            "kmeans",
            "cluster",
            "pca",
            "ica",
            "tsne",
            "dbscan",
            "isolation",
            "anomaly",
            "outlier",
            "decomposition",
            # Unsupervised preprocessing
            "standardscaler",
            "minmaxscaler",
            "robustscaler",
            "normalizer",
            "quantiletransformer",
            "powertransformer",
            "polynomialfeatures",
            "onehotencoder",
            "labelencoder",
            "imputer",
            "simpleimputer",
        ]

        for pattern in supervised_patterns:
            if pattern in class_name:
                return "supervised"

        for pattern in unsupervised_patterns:
            if pattern in class_name:
                return "unsupervised"
        return None

    def check_module_origin():
        module = instance.__class__.__module__
        if module:
            if "cluster" in module or "decomposition" in module:
                return "unsupervised"
            elif (
                "ensemble" in module
                or "linear_model" in module
                or "svm" in module
                or "feature_selection" in module
            ):
                return "supervised"
            elif "preprocessing" in module:
                # Most sklearn preprocessing is unsupervised (scalers, encoders)
                # but feature selection is supervised
                if "feature_selection" in module:
                    return "supervised"
                else:
                    return "unsupervised"
        return None

    def check_model_attributes():
        supervised_attrs = [
            "classes_",
            "n_classes_",
            "coef_",
            "intercept_",
            "feature_importances_",
            "estimators_",
            # Feature selection attributes
            "scores_",
            "pvalues_",
            "support_",
            "ranking_",
        ]

        unsupervised_attrs = [
            "cluster_centers_",
            "labels_",
            "inertia_",
            "components_",
            "explained_variance_",
            "singular_values_",
            "n_clusters",
            # Preprocessing attributes
            "scale_",
            "mean_",
            "var_",
            "min_",
            "max_",
            "data_min_",
            "data_max_",
            "categories_",
        ]

        supervised_count = sum(
            1 for attr in supervised_attrs if hasattr(instance, attr)
        )
        unsupervised_count = sum(
            1 for attr in unsupervised_attrs if hasattr(instance, attr)
        )

        if supervised_count > unsupervised_count:
            return "supervised"
        elif unsupervised_count > supervised_count:
            return "unsupervised"
        return None

    strategies = [
        check_method_patterns,
        check_class_name,
        check_module_origin,
        check_model_attributes,
    ]

    votes = {"supervised": 0, "unsupervised": 0}

    for strategy in strategies:
        try:
            result = strategy()
            if result in votes:
                votes[result] += 1
        except Exception:
            continue

    if votes["supervised"] > votes["unsupervised"]:
        return "supervised"
    elif votes["unsupervised"] > votes["supervised"]:
        return "unsupervised"
    else:
        return "supervised"


def execute_fit(instance, x, y):
    instance_type = get_instance_type(instance)
    if instance_type == "supervised":
        instance.fit(x, y)
    elif instance_type == "unsupervised":
        instance.fit(x)
    return instance


def serialize(pipeline, filepath):
    serialize_graph(pipeline._PipelineBase__graph, filepath)


def load(directory):
    raise NotImplementedError("Loading from directory is not implemented yet.")
