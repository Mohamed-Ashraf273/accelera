class MetricWrapper:
    def __init__(self, metric, binary_proba, **params):
        self.metric = metric
        self.params = params
        self.binary_proba = binary_proba

    def execute(self, y_true, y_pred):
        y_pred = (
            self.__handel_binary_proba(y_pred, y_true)
            if self.binary_proba
            else y_pred
        )
        return self.metric(y_true, y_pred, **self.params)

    def __handel_binary_proba(self, y_pred, y_true):
        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
            return y_pred
        return y_pred
