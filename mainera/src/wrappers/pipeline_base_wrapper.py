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

    def set_multicore_threshold(self, threshold):
        self.__graph.setMulticoreThreshold(threshold)
        return self

    def disable_parallel_execution(self):
        self.__graph.enableParallelExecution(False)
        return self

    def save_preprocessed_data(self, directory):
        if not self.__graph.savePreprocessedData(directory):
            raise ValueError("Saving data to disk failed.")
        return self
