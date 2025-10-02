import logging
from unittest.mock import patch

import pytest

from mainera.src.utils.mainera_utils import get_metric_object, get_correct_metric_class
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


# test for get_correct_metric_class
def supervised_metric_test1(y_true, y_pred):
    return 0


def supervised_metric_test2(y_true, y_proba):
    return 1


def supervised_metric_test3(y_true, y_score):
    return 2


def supervised_metric_test4(labels_true, labels_pred):
    return 3


def unsupervised_metric(X, labels):
    return 4


def unvalid_metric(a, b, c):
    return 5


class DummySupervisedWrapper:
    def __init__(self, metric_name, metric, y_true, **params):
        self.metric_name = metric_name
        self.metric = metric
        self.y_true = y_true
        self.params = params


class DummyUnSupervisedWrapper:
    def __init__(self, metric_name, metric, X, **params):
        self.metric_name = metric_name
        self.metric = metric
        self.X = X
        self.params = params


@pytest.mark.parametrize(
    "metric, expected_class",
    [
        (supervised_metric_test1, DummySupervisedWrapper),
        (supervised_metric_test2, DummySupervisedWrapper),
        (supervised_metric_test3, DummySupervisedWrapper),
        (supervised_metric_test4, DummySupervisedWrapper),
        (unsupervised_metric, DummyUnSupervisedWrapper),
        (unvalid_metric, None),
    ],
)
def test_get_correct_metric_class(metric, expected_class):
    with (
        patch(
            "mainera.src.utils.mainera_utils.SupervisedMetricWrapper",
            DummySupervisedWrapper,
        ),
        patch(
            "mainera.src.utils.mainera_utils.UnSupervisedMetricWrapper",
            DummyUnSupervisedWrapper,
        ),
    ):
        metric_name = "test_metric"
        y_true = [0, 1, 1, 0]
        X = [[1], [2], [3], [4]]
        params = {"param1": "1"}

        result = get_correct_metric_class(
            metric_name, metric, y_true=y_true, X=X, **params
        )

        if expected_class is None:
            assert result is None
        else:
            assert isinstance(result, expected_class)
            assert result.metric_name == metric_name
            assert result.metric == metric
            assert result.params == params
            if expected_class == DummySupervisedWrapper:
                assert result.y_true == y_true
            if expected_class == DummyUnSupervisedWrapper:
                assert result.X == X
