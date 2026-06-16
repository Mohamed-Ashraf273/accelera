import pickle
from pathlib import Path

from accelera.src.config import config

try:
    import graph
except ImportError as e:
    raise ImportError(
        "The 'graph' C++ module could not be imported. "
        "Please ensure it is built and available in your PYTHONPATH."
    ) from e


class PipelineBase:
    def __init__(self, _graph=None):
        self.__graph = _graph if _graph is not None else graph.Graph()
        self.__graph.enableParallelExecution(True)
        self.types = {
            "preprocess": graph.NodeType.PREPROCESS,
            "model": graph.NodeType.MODEL,
            "predict": graph.NodeType.PREDICT,
            "metric": graph.NodeType.METRIC,
            "merge": graph.NodeType.MERGE,
        }

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "__call__ method must be implemented in subclasses."
        )

    @staticmethod
    def _resolve_pipeline_path(path):
        pipeline_path = Path(path)
        if pipeline_path.exists() and pipeline_path.is_dir():
            return pipeline_path / config.PIPELINE_FILENAME
        return pipeline_path

    def set_multicore_threshold(self, threshold):
        self.__graph.setMulticoreThreshold(threshold)
        return self

    def disable_parallel_execution(self):
        self.__graph.enableParallelExecution(False)
        return self

    def save(self, path=config.PIPELINE_FILENAME):
        pipeline_path = self._resolve_pipeline_path(path)
        if pipeline_path.parent != Path("."):
            pipeline_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pipeline_path, "wb") as file:
            pickle.dump(self, file)

        return self

    @classmethod
    def load(cls, path=config.PIPELINE_FILENAME):
        pipeline_path = cls._resolve_pipeline_path(path)

        with open(pipeline_path, "rb") as file:
            loaded_pipeline = pickle.load(file)

        if not isinstance(loaded_pipeline, cls):
            raise TypeError(
                f"Expected saved {cls.__name__}, got "
                f"{type(loaded_pipeline).__name__}"
            )

        return loaded_pipeline
