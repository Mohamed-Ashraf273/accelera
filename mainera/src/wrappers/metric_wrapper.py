class MetricWrapper:
    def __init__(self, metric, **params):
        self.metric = metric
        self.params = params

    def execute(self, y_true, y_pred):
        return self.metric(y_true, y_pred, **self.params)
