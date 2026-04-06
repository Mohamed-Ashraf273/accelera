import os

from accelera.src.accelera_pipe.core.pipeline_base import PipelineBase
from accelera.src.utils.accelera_utils import serialize


class ExecutedGraph(PipelineBase):
    def __init__(self, executed_graph):
        super().__init__(_graph=executed_graph)

    def __call__(self, X, y_true=None):
        if y_true is not None:
            self._PipelineBase__graph.enableDisableMetrics(
                y_true=y_true, enable=True
            )

        results = self._PipelineBase__graph.execute(X)

        if y_true is not None:
            self._PipelineBase__graph.enableDisableMetrics(enable=False)

        return results

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        serialize(self, os.path.join(directory, "pipeline.xml"))
        # TODO: we need also to save models and other
        # data needed later for loading
        return self
