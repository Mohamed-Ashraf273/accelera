from abc import ABC
from abc import abstractmethod

import numpy as np

# TODO (1): make each class in a file


def convert_to_array(x):  # TODO (2): move to mainera_utils
    if x is None:
        return None
    return np.asarray(x)


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

    def _validate_shape(
        self, y_pred, expected_shape, names
    ):  # TODO (3): move to mainera_utils and
        # make it more general (array, expected_shape, names=None)
        if y_pred.shape[0] != expected_shape:
            raise ValueError(
                f"{names[0]} , {names[1]} must have the same "
                f"number of samples. "
                f"Got {names[0]} has {y_pred.shape[0]} "
                f"samples and {names[1]} has {expected_shape} samples."
            )


class SupervisedMetricWrapper(BaseMetricWrapper):
    def __init__(
        self,
        metric_name,
        metric,
        y_true=None,
        X=None,
        **params,
    ):
        super().__init__(
            metric_name,
            metric,
            y_true=y_true,
            X=X,
            **params,
        )
        self.__need_1d_probability = [
            "roc_auc_score",
            "average_precision_score",
            "brier_score_loss",
            "roc_curve",
            "precision_recall_curve",
        ]
        print("supervised metric initialized")

    def execute(self, y_pred):
        y_pred = convert_to_array(y_pred)
        if self.y_true is None:
            raise ValueError("y_true must be provided for supervised metrics.")
        self._validate_shape(y_pred, self.y_true.shape[0], ["y_pred", "y_true"])
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


class UnSupervisedMetricWrapper(BaseMetricWrapper):
    def __init__(
        self,
        metric_name,
        metric,
        y_true=None,
        X=None,
        **params,
    ):
        super().__init__(
            metric_name,
            metric,
            y_true=y_true,
            X=X,
            **params,
        )
        print("unsupervised metric initialized")

    def execute(self, y_pred):
        y_pred = convert_to_array(y_pred)

        if self.X is None:
            raise ValueError("X must be provided for unsupervised metrics.")
        self._validate_shape(y_pred, self.X.shape[0], ["y_pred", "X"])

        result = self.metric(self.X, y_pred, **self.params)
        output = {"metric name": self.metric_name, "result": result}
        return output
