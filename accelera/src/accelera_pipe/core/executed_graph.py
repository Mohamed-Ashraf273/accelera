from accelera.src.accelera_pipe.core.pipeline_base import PipelineBase


class ExecutedGraph(PipelineBase):
    def __init__(self, executed_graph):
        super().__init__(_graph=executed_graph)

    def __call__(self, X, y_true=None):
        if y_true is not None:
            self._PipelineBase__graph.enableDisableMetrics(
                y_true=y_true, enable=True
            )

        results = self.predict(X)

        if y_true is not None:
            self._PipelineBase__graph.enableDisableMetrics(enable=False)

        return results

    def predict(self, X):
        results = self._PipelineBase__graph.execute(X)
        return results
