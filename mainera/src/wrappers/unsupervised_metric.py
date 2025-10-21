from mainera.src.core.metric import BaseMetric
from mainera.src.utils.array_utils import convert_to_array
from mainera.src.utils.array_utils import validate_array_shape


class UnSupervisedMetric(BaseMetric):
    def __init__(
        self,
        metric_name,
        metric,
        plot_func=None,
        labels_name=None,
        headers_name=None,
        **params,
    ):
        super().__init__(
            metric_name,
            metric,
            y_true=None,
            plot_func=plot_func,
            labels_name=labels_name,
            headers_name=headers_name,
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

        output = {
            "metric name": self.metric_name,
            "result": result,
            "plot_func": self.plot_func,
            "labels_name": self.labels_name,
            "headers_name": self.headers_name,
        }
        return output
