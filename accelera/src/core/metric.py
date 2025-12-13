from abc import ABC
from abc import abstractmethod

from accelera.src.utils.array_utils import convert_to_array


class BaseMetric(ABC):
    def __init__(
        self,
        metric_name,
        metric,
        y_true=None,
        X=None,
        plot_func=None,
        labels_name=None,
        headers_name=None,
        **params,
    ):
        self.metric = metric
        self.params = params
        self.metric_name = metric_name
        self.y_true = convert_to_array(y_true)
        self.X = convert_to_array(X)
        self.plot_func = plot_func
        self.labels_name = labels_name
        self.headers_name = headers_name

    @abstractmethod
    def execute(self, y_pred):
        pass
