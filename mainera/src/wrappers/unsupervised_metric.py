from mainera.src.core.metric import BaseMetric
from mainera.src.utils.array_utils import convert_to_array
from mainera.src.utils.array_utils import validate_array_shape


class UnSupervisedMetric(BaseMetric):
    def __init__(
        self,
        metric_name,
        metric,
        tuple_argums=None,
        labels_name=None,
        **params,
    ):
        super().__init__(
            metric_name,
            metric,
            y_true=None,
            tuple_argums=tuple_argums,
            labels_name=labels_name,
            **params,
        )
        self.X = None

    def set_X(self, X):
        self.X = convert_to_array(X)

    def execute(self, y_pred):
        y_pred = convert_to_array(y_pred)

        if self.X is None:
            raise ValueError("X must be provided for unsupervised metrics.")
        validate_array_shape(y_pred, self.X.shape[0], ["y_pred", "X"])

        result = self.metric(self.X, y_pred, **self.params)
        self.check_tuple_argums(result)

        output = {
            "metric name": self.metric_name,
            "result": result,
            "tuple_argums": self.tuple_argums,
            "labels_name": self.labels_name,
        }
        return output
