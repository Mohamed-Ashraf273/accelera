from abc import ABC
from abc import abstractmethod

from mainera.src.utils.array_utils import convert_to_array


class BaseMetricWrapper(ABC):
    def __init__(
        self,
        metric_name,
        metric,
        y_true=None,
        X=None,
        **params,
    ):
        self.metric = metric
        self.params = params
        self.metric_name = metric_name
        self.y_true = convert_to_array(y_true)
        self.X = convert_to_array(X)

    @abstractmethod
    def execute(self, y_pred):
        pass
