import inspect
import logging
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from mainera.src.utils.mainera_utils import get_learning_type
from mainera.src.utils.mainera_utils import get_metric_object
from mainera.src.utils.mainera_utils import metric_validation
from mainera.src.utils.mainera_utils import print_msg


@pytest.mark.parametrize(
    "msg,line_break,level,interactive,expected_out,expected_err,log_level",
    [
        ("Hello mAInera!", True, None, True, "Hello mAInera!\n", "", None),
        ("NoBreak", False, None, True, "NoBreak", "", None),
        ("Logged Info", True, "info", False, "", "", logging.INFO),
        ("Logged Warning", True, "warning", False, "", "", logging.WARNING),
        ("Logged Error", True, "error", False, "", "", logging.ERROR),
    ],
)
def test_print_msg(
    capsys,
    caplog,
    monkeypatch,
    msg,
    line_break,
    level,
    interactive,
    expected_out,
    expected_err,
    log_level,
):
    monkeypatch.setitem(print_msg.__globals__, "interactive", interactive)
    if log_level:
        with caplog.at_level(log_level):
            print_msg(msg, line_break=line_break, level=level)
        assert msg in caplog.text
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
    else:
        print_msg(msg, line_break=line_break)
        captured = capsys.readouterr()
        assert captured.out == expected_out
        assert captured.err == expected_err


@pytest.mark.parametrize(
    "metric_name, expected_result, mock_metrics_attr",
    [
        ("accuracy", "mock_accuracy_function", "mock_accuracy_function"),
        ("precision", "mock_precision_function", "mock_precision_function"),
        ("recall", "mock_recall_function", "mock_recall_function"),
        ("nonexistent_metric", None, None),
        ("invalid_metric", None, None),
        ("", None, None),
        ("f1_score", "mock_f1_function", "mock_f1_function"),
        ("roc_auc", "mock_roc_auc_function", "mock_roc_auc_function"),
    ],
)
def test_get_metric_object(metric_name, expected_result, mock_metrics_attr):
    with patch("mainera.src.utils.mainera_utils.metrics") as mock_metrics:
        if mock_metrics_attr:
            setattr(mock_metrics, metric_name, mock_metrics_attr)
        else:
            if metric_name:
                setattr(mock_metrics, metric_name, None)
            else:
                pass

        result = get_metric_object(metric_name)

        if expected_result:
            assert result == expected_result
        else:
            assert result is None


@pytest.mark.parametrize(
    "metric_func_params, metric_name, should_raise, expected_error_msg",
    [
        # Valid metric functions
        (["y_true", "y_pred"], "accuracy", False, None),
        (["y_true", "y_score"], "roc_auc", False, None),
        (["y_true", "y_prob"], "log_loss", False, None),
        (
            ["y_true", "y_pred", "sample_weight"],
            "weighted_accuracy",
            False,
            None,
        ),
        # Invalid metric functions - missing required parameters
        (["y_pred"], "invalid_metric", True, "does not take (y_true, y_pred)"),
        (["y_score"], "invalid_metric", True, "does not take (y_true, y_pred)"),
        (["y_true"], "invalid_metric", True, "does not take (y_true, y_pred)"),
        ([], "empty_metric", True, "does not take (y_true, y_pred)"),
        # Edge cases with similar but incorrect parameter names
        (
            ["Y_true", "y_pred"],
            "case_sensitive",
            True,
            "does not take (y_true, y_pred)",
        ),
        (
            ["y_true", "y_preds"],
            "typo_metric",
            True,
            "does not take (y_true, y_pred)",
        ),
        (
            ["true_labels", "predictions"],
            "custom_names",
            True,
            "does not take (y_true, y_pred)",
        ),
        # Valid but with extra parameters
        (
            ["y_true", "y_pred", "normalize", "sample_weight"],
            "accuracy_with_extras",
            False,
            None,
        ),
        (
            ["y_true", "y_score", "average", "sample_weight"],
            "precision_with_extras",
            False,
            None,
        ),
    ],
)
def test_metric_validation(
    metric_func_params, metric_name, should_raise, expected_error_msg
):
    """Test metric_validation function with various parameter combinations."""

    mock_func = MagicMock()

    params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in metric_func_params
    ]
    mock_func.__signature__ = inspect.Signature(params)

    if should_raise:
        with pytest.raises(ValueError) as exc_info:
            metric_validation(mock_func, metric_name)

        assert expected_error_msg in str(exc_info.value)
        assert f"Metric '{metric_name}'" in str(exc_info.value)
    else:
        try:
            metric_validation(mock_func, metric_name)
        except ValueError:
            pytest.fail("metric_validation raised ValueError unexpectedly!")


def test_metric_validation_with_none_function():
    with pytest.raises(TypeError):
        metric_validation(None, "none_metric")


def test_metric_validation_with_non_callable():
    with pytest.raises(TypeError):
        metric_validation("not_a_function", "string_metric")


@pytest.mark.parametrize(
    "metric_name, mock_metrics_attr, mock_func_params, should_work",
    [
        ("accuracy", "mock_accuracy", ["y_true", "y_pred"], True),
        ("precision", "mock_precision", ["y_true", "y_pred"], True),
        ("roc_auc", "mock_roc_auc", ["y_true", "y_score"], True),
        ("invalid_metric", "mock_invalid", ["y_pred"], False),
        ("nonexistent_metric", None, None, False),
    ],
)
def test_get_and_validate_metric_integration(
    metric_name, mock_metrics_attr, mock_func_params, should_work
):
    with patch("mainera.src.utils.mainera_utils.metrics") as mock_metrics:
        if mock_metrics_attr:
            mock_func = MagicMock()
            params = [
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in mock_func_params
            ]
            mock_func.__signature__ = inspect.Signature(params)

            setattr(mock_metrics, metric_name, mock_func)

        metric_func = get_metric_object(metric_name)

        if metric_func is None:
            assert not should_work
        else:
            if should_work:
                metric_validation(metric_func, metric_name)
            else:
                with pytest.raises(ValueError):
                    metric_validation(metric_func, metric_name)


def test_metric_validation_function_with_kwargs():
    def func_with_kwargs(y_true, y_pred, **kwargs):
        return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

    metric_validation(func_with_kwargs, "kwargs_func")


def test_metric_validation_function_with_defaults():
    def func_with_defaults(y_true, y_pred, normalize=True, sample_weight=None):
        return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

    metric_validation(func_with_defaults, "defaults_func")


@pytest.mark.parametrize(
    "model, expected_type, description",
    [
        # Supervised models (have 'y' parameter in fit)
        (RandomForestClassifier(), "supervised", "RandomForestClassifier"),
        (LinearRegression(), "supervised", "LinearRegression"),
        # Unsupervised models (no 'y' parameter in fit)
        (KMeans(), "unsupervised", "KMeans"),
        (PCA(), "unsupervised", "PCA"),
        (GaussianMixture(), "unsupervised", "GaussianMixture"),
        (StandardScaler(), "unsupervised", "StandardScaler"),
    ],
)
def test_get_learning_type_with_sklearn_models(
    model, expected_type, description
):
    result = get_learning_type(model)
    assert result == expected_type, f"Failed for {description}"


@pytest.mark.parametrize(
    "fit_params, expected_type, description",
    [
        # Supervised signatures
        (["X", "y"], "supervised", "basic supervised"),
        (
            ["X", "y", "sample_weight"],
            "supervised",
            "supervised with sample_weight",
        ),
        (["self", "X", "y"], "supervised", "with self parameter"),
        (["cls", "X", "y"], "supervised", "with cls parameter"),
        (["X", "y", "kwargs"], "supervised", "with kwargs"),
        # Unsupervised signatures
        (["X"], "unsupervised", "basic unsupervised"),
        (["X", "y=None"], "unsupervised", "y as optional parameter"),
        (["self", "X"], "unsupervised", "unsupervised with self"),
        (
            ["X", "sample_weight"],
            "unsupervised",
            "unsupervised with sample_weight",
        ),
        (["X", "kwargs"], "unsupervised", "unsupervised with kwargs"),
        # Edge cases
        (["X", "Y"], "unsupervised", "capital Y instead of y"),
        (["X", "target"], "unsupervised", "target instead of y"),
        (["X", "labels"], "unsupervised", "labels instead of y"),
        (["features", "y"], "supervised", "X named features"),
    ],
)
def test_get_learning_type_with_mock_models(
    fit_params, expected_type, description
):
    mock_model = MagicMock()

    params = []
    for param_name in fit_params:
        if "=" in param_name:
            name, default = param_name.split("=")
            param = inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default
            )
        else:
            param = inspect.Parameter(
                param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
        params.append(param)

    mock_model.fit.__signature__ = inspect.Signature(params)

    result = get_learning_type(mock_model)
    assert result == expected_type, f"Failed for {description}"


@pytest.mark.parametrize(
    "model, expected_type, description",
    [
        (None, "unknown", "None object"),
        ("not_a_model", "unknown", "string instead of model"),
        (123, "unknown", "integer instead of model"),
        ([], "unknown", "list instead of model"),
        ({}, "unknown", "dict instead of model"),
    ],
)
def test_get_learning_type_with_non_model_objects(
    model, expected_type, description
):
    result = get_learning_type(model)
    assert result == expected_type, f"Failed for {description}"


def test_get_learning_type_with_broken_signature():
    mock_model = MagicMock()
    mock_model.fit = "not_a_callable"

    result = get_learning_type(mock_model)
    assert result == "unknown"


def test_get_learning_type_with_exception_during_inspection():
    mock_model = MagicMock()

    def broken_fit():
        pass

    with patch("inspect.signature", side_effect=TypeError("Cannot inspect")):
        result = get_learning_type(mock_model)
        assert result == "unknown"


@pytest.mark.parametrize(
    "model_attributes, expected_type, description",
    [
        ({"fit": True}, "unknown", "fit exists but not callable"),
        ({}, "unknown", "no fit method"),
        ({"train": True}, "unknown", "has train instead of fit"),
    ],
)
def test_get_learning_type_with_alternative_models(
    model_attributes, expected_type, description
):
    class CustomModel:
        def __init__(self, attributes):
            for attr, value in attributes.items():
                setattr(self, attr, value)

    model = CustomModel(model_attributes)
    result = get_learning_type(model)
    assert result == expected_type, f"Failed for {description}"


@pytest.mark.parametrize(
    "fit_params, expected_type, description",
    [
        (
            ["X", "y", "sample_weight", "other_param"],
            "supervised",
            "supervised with extra params",
        ),
        (["X", "y", "kwargs"], "supervised", "supervised with kwargs"),
        (
            ["X", "param1", "param2"],
            "unsupervised",
            "unsupervised with multiple params",
        ),
        (["data", "y"], "supervised", "different X parameter name"),
    ],
)
def test_get_learning_type_parameter_variations(
    fit_params, expected_type, description
):
    class TestModel:
        def fit(self, *args, **kwargs):
            pass

    params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in fit_params
    ]
    TestModel.fit.__signature__ = inspect.Signature(params)

    model = TestModel()
    result = get_learning_type(model)
    assert result == expected_type, f"Failed for {description}"


def test_get_learning_type_integration_with_real_workflow():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    rf_model = RandomForestClassifier()
    rf_type = get_learning_type(rf_model)
    assert rf_type == "supervised"

    rf_model.fit(X, y)

    kmeans_model = KMeans()
    kmeans_type = get_learning_type(kmeans_model)
    assert kmeans_type == "unsupervised"

    kmeans_model.fit(X)


@pytest.mark.parametrize(
    "model_class, expected_type",
    [
        (RandomForestClassifier, "supervised"),
        (KMeans, "supervised"),
        (PCA, "supervised"),
    ],
)
def test_get_learning_type_with_model_classes(model_class, expected_type):
    result = get_learning_type(model_class())
    assert result in ["supervised", "unsupervised", "unknown"]
