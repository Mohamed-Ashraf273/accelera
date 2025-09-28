import inspect
import logging
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

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
        (["y_true", "y_proba"], "log_loss", False, None),
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
