import logging
from unittest.mock import patch

import pytest

from mainera.src.utils.mainera_utils import get_metric_object
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
