import pickle
from pathlib import Path

from accelera.src.accelera_pipe.core.pipeline_base import PipelineBase
from accelera.src.config import config


def _resolve_pipeline_path(path):
    pipeline_path = Path(path)
    if pipeline_path.exists() and pipeline_path.is_dir():
        return pipeline_path / config.PIPELINE_FILENAME
    return pipeline_path


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

    def save(self, path=config.PIPELINE_FILENAME):
        pipeline_path = _resolve_pipeline_path(path)
        if pipeline_path.parent != Path("."):
            pipeline_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pipeline_path, "wb") as file:
            pickle.dump(self, file)

        return self

    @classmethod
    def load(cls, path=config.PIPELINE_FILENAME):
        pipeline_path = _resolve_pipeline_path(path)

        with open(pipeline_path, "rb") as file:
            loaded_pipeline = pickle.load(file)

        if not isinstance(loaded_pipeline, cls):
            raise TypeError(
                f"Expected saved {cls.__name__}, got "
                f"{type(loaded_pipeline).__name__}"
            )

        return loaded_pipeline
