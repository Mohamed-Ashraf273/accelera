import logging
from unittest.mock import MagicMock
from unittest.mock import patch

from accelera.src.utils.mainera_utils import get_correct_metric_class
from accelera.src.utils.mainera_utils import get_instance_type
from accelera.src.utils.mainera_utils import get_metric_object
from accelera.src.utils.mainera_utils import print_msg


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
                self,
                metric_name,
                metric,
                y_true,
                plot_func,
                labels_name,
                headers_name,
                **params,
            ):
                self.metric_name = metric_name
                self.metric = metric
                self.y_true = y_true
                self.plot_func = plot_func
                self.params = params
                self.labels_name = labels_name
                self.headers_name = headers_name

        with patch(
            "mainera.src.utils.mainera_utils.SupervisedMetric",
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
            def __init__(
                self,
                metric_name,
                metric,
                plot_func,
                labels_name,
                headers_name,
                **params,
            ):
                self.metric_name = metric_name
                self.metric = metric
                self.params = params
                self.plot_func = plot_func
                self.labels_name = labels_name
                self.headers_name = headers_name

        with patch(
            "mainera.src.utils.mainera_utils.UnSupervisedMetric",
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


class TestGetInstanceType:
    def test_supervised_algorithms(self):
        clf = MagicMock()
        clf.__class__.__name__ = "RandomForestClassifier"
        clf.__class__.__module__ = "sklearn.ensemble"
        assert get_instance_type(clf) == "supervised"

        reg = MagicMock()
        reg.__class__.__name__ = "LinearRegression"
        reg.__class__.__module__ = "sklearn.linear_model"
        assert get_instance_type(reg) == "supervised"

    def test_unsupervised_algorithms(self):
        kmeans = MagicMock()
        kmeans.__class__.__name__ = "KMeans"
        kmeans.__class__.__module__ = "sklearn.cluster"
        assert get_instance_type(kmeans) == "unsupervised"

        pca = MagicMock()
        pca.__class__.__name__ = "PCA"
        pca.__class__.__module__ = "sklearn.decomposition"
        assert get_instance_type(pca) == "unsupervised"

    def test_feature_selection_supervised(self):
        selector = MagicMock()
        selector.__class__.__name__ = "SelectKBest"
        selector.__class__.__module__ = "sklearn.feature_selection"
        assert get_instance_type(selector) == "supervised"

    def test_preprocessing_unsupervised(self):
        scaler = MagicMock()
        scaler.__class__.__name__ = "StandardScaler"
        scaler.__class__.__module__ = "sklearn.preprocessing"
        assert get_instance_type(scaler) == "unsupervised"

    def test_real_sklearn_objects(self):
        try:
            from sklearn.cluster import KMeans
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import SelectKBest
            from sklearn.preprocessing import StandardScaler

            assert get_instance_type(RandomForestClassifier()) == "supervised"
            assert get_instance_type(KMeans(n_clusters=2)) == "unsupervised"
            assert get_instance_type(SelectKBest()) == "supervised"
            assert get_instance_type(StandardScaler()) == "unsupervised"
        except ImportError:
            pass  # Skip if sklearn not available

    def test_unknown_defaults_to_supervised(self):
        class Unknown:
            pass

        assert get_instance_type(Unknown()) == "supervised"
