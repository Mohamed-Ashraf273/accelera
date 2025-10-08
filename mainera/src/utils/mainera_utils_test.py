import logging
from unittest.mock import patch

from mainera.src.utils.mainera_utils import get_correct_metric_class
from mainera.src.utils.mainera_utils import get_metric_object
from mainera.src.utils.mainera_utils import print_msg


class TestPrintMsg:
    def test_print_msg_interactive_with_line_break(self, capsys, monkeypatch):
        monkeypatch.setitem(print_msg.__globals__, "interactive", True)
        print_msg("Hello mAInera!")
        captured = capsys.readouterr()
        assert captured.out == "Hello mAInera!\n"

    def test_print_msg_interactive_without_line_break(
        self, capsys, monkeypatch
    ):
        monkeypatch.setitem(print_msg.__globals__, "interactive", True)
        print_msg("NoBreak", line_break=False)
        captured = capsys.readouterr()
        assert captured.out == "NoBreak"

    def test_print_msg_logging_mode(self, caplog, capsys, monkeypatch):
        monkeypatch.setitem(print_msg.__globals__, "interactive", False)
        with caplog.at_level(logging.INFO):
            print_msg("Logged Info", level="info")
        assert "Logged Info" in caplog.text
        captured = capsys.readouterr()
        assert captured.out == ""


class TestGetMetricObject:
    def test_get_existing_metric(self):
        with patch("mainera.src.utils.mainera_utils.metrics") as mock_metrics:
            mock_metrics.accuracy = "mock_accuracy_function"
            result = get_metric_object("accuracy")
            assert result == "mock_accuracy_function"

    def test_get_nonexistent_metric(self):
        with patch("mainera.src.utils.mainera_utils.metrics") as mock_metrics:
            mock_metrics.nonexistent_metric = None
            result = get_metric_object("nonexistent_metric")
            assert result is None

    def test_get_empty_metric_name(self):
        result = get_metric_object("")
        assert result is None


class TestGetCorrectMetricClass:
    def test_supervised_metric_detection(self):
        def supervised_metric(y_true, y_pred):
            return 0

        class DummySupervisedWrapper:
            def __init__(
                self, metric_name, metric, y_true, tuple_argums, **params
            ):
                self.metric_name = metric_name
                self.metric = metric
                self.y_true = y_true
                self.tuple_argums = tuple_argums
                self.params = params

        with patch(
            "mainera.src.utils.mainera_utils.SupervisedMetricWrapper",
            DummySupervisedWrapper,
        ):
            result = get_correct_metric_class(
                "test_metric", supervised_metric, y_true=[0, 1, 1, 0]
            )
            assert isinstance(result, DummySupervisedWrapper)
            assert result.metric_name == "test_metric"

    def test_unsupervised_metric_detection(self):
        def unsupervised_metric(X, labels):
            return 0

        class DummyUnSupervisedWrapper:
            def __init__(self, metric_name, metric, tuple_argums, **params):
                self.metric_name = metric_name
                self.metric = metric
                self.params = params
                self.tuple_argums = tuple_argums

        with patch(
            "mainera.src.utils.mainera_utils.UnSupervisedMetricWrapper",
            DummyUnSupervisedWrapper,
        ):
            result = get_correct_metric_class(
                "test_metric", unsupervised_metric, X=[[1], [2], [3]]
            )
            assert isinstance(result, DummyUnSupervisedWrapper)
            assert result.metric_name == "test_metric"

    def test_invalid_metric_returns_none(self):
        def invalid_metric(a, b, c):
            return 0

        result = get_correct_metric_class("test_metric", invalid_metric)
        assert result is None
