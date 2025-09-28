import numpy as np


class MetricWrapper:
    def __init__(self, metric_name, metric, **params):
        self.metric = metric
        self.params = params
        self.metric_name = metric_name
        self.__need_1d_probability = [
            "roc_auc_score",
            "average_precision_score",
            "brier_score_loss",
            "roc_curve",
            "precision_recall_curve",
        ]

    def execute(self, y_true, y_pred):
        y_pred = self.__handel_binary_proba(y_pred)
        result = self.metric(y_true, y_pred, **self.params)
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
