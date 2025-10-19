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
        tuple_argums=None,
        labels_name=None,
        **params,
    ):
        self.metric = metric
        self.params = params
        self.metric_name = metric_name
        self.y_true = convert_to_array(y_true)
        self.X = convert_to_array(X)
        self.tuple_argums = tuple_argums
        self.labels_name = labels_name

    def check_tuple_argums(self, result):
        if isinstance(result, tuple):
            if (
                self.tuple_argums is None
                or self.tuple_argums["is_curve"] is None
            ):
                raise ValueError(
                    "tuple_argums must be an object contains keys 'is_curve'"
                )
            if not self.tuple_argums["is_curve"]:
                if "item_name" not in self.tuple_argums:
                    raise ValueError(
                        "tuple_argums must be an object "
                        "contains keys 'item_name'"
                    )
                if len(self.tuple_argums["item_name"]) != len(result):
                    raise ValueError(
                        "the length of item_name must be equal to the "
                        "length of result the length "
                        f"of result is {len(result)}"
                    )
            else:
                if "plot_func" not in self.tuple_argums:
                    raise ValueError(
                        "tuple_argums must be an object contains keys "
                        "'plot_func' this function is for plotting the result "
                        "if the metric return a curve"
                    )

    @abstractmethod
    def execute(self, y_pred):
        pass
