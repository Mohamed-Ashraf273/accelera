from mainera.src.core.metric import BaseMetric
from mainera.src.utils.array_utils import convert_to_array
from mainera.src.utils.array_utils import validate_array_shape


class SupervisedMetric(BaseMetric):
    def __init__(
        self,
        metric_name,
        metric,
        y_true=None,
        tuple_argums=None,
        labels_name=None,
        **params,
    ):
        super().__init__(
            metric_name,
            metric,
            y_true=y_true,
            X=None,
            tuple_argums=tuple_argums,
            labels_name=labels_name,
            **params,
        )

    def set_y_true(self, new_y_true):
        self.y_true = convert_to_array(new_y_true)

    def execute(self, y_pred):
        y_pred = convert_to_array(y_pred)
        if self.y_true is None:
            raise ValueError("y_true must be provided for supervised metrics.")
        validate_array_shape(y_pred, self.y_true.shape[0], ["y_pred", "y_true"])
        result = self.metric(self.y_true, y_pred, **self.params)
        self.check_tuple_argums(result)
        output = {
            "metric name": self.metric_name,
            "result": result,
            "tuple_argums": self.tuple_argums,
            "labels_name": self.labels_name,
        }
        return output
