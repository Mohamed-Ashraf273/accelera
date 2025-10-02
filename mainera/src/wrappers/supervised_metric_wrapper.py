from mainera.src.utils.array_utils import convert_to_array
from mainera.src.utils.array_utils import validate_array_shape
from mainera.src.wrappers.metric_wrapper import BaseMetricWrapper


class SupervisedMetricWrapper(BaseMetricWrapper):
    def __init__(
        self,
        metric_name,
        metric,
        y_true=None,
        **params,
    ):
        super().__init__(
            metric_name,
            metric,
            y_true=y_true,
            X=None,
            **params,
        )
        self.__need_1d_probability = [
            "roc_auc_score",
            "average_precision_score",
            "brier_score_loss",
            "roc_curve",
            "precision_recall_curve",
        ]

    def execute(self, y_pred):
        y_pred = convert_to_array(y_pred)
        if self.y_true is None:
            raise ValueError("y_true must be provided for supervised metrics.")
        validate_array_shape(y_pred, self.y_true.shape[0], ["y_pred", "y_true"])
        y_pred = self.__handel_binary_proba(y_pred)
        result = self.metric(self.y_true, y_pred, **self.params)
        output = {"metric name": self.metric_name, "result": result}
        return output

    def __handel_binary_proba(self, y_pred):
        if (
            self.metric_name in self.__need_1d_probability
            and y_pred.ndim == 2
            and y_pred.shape[1] == 2
        ):
            y_pred = y_pred[:, 1]
            return y_pred
        return y_pred
