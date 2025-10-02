from mainera.src.utils.array_utils import convert_to_array
from mainera.src.utils.array_utils import validate_array_shape
from mainera.src.wrappers.metric_wrapper import BaseMetricWrapper


class UnSupervisedMetricWrapper(BaseMetricWrapper):
    def __init__(
        self,
        metric_name,
        metric,
        X=None,
        **params,
    ):
        super().__init__(
            metric_name,
            metric,
            y_true=None,
            X=X,
            **params,
        )

    def execute(self, y_pred):
        y_pred = convert_to_array(y_pred)

        if self.X is None:
            raise ValueError("X must be provided for unsupervised metrics.")
        validate_array_shape(y_pred, self.X.shape[0], ["y_pred", "X"])

        result = self.metric(self.X, y_pred, **self.params)
        output = {"metric name": self.metric_name, "result": result}
        return output
