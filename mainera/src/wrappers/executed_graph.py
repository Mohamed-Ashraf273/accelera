class ExecutedGraphWrapper:
    def __init__(self, executed_graph):
        self.__executed_graph = executed_graph
        self.__executed_graph.enableParallelExecution(True)

    def __call__(self, X, y, metrics=False, y_true=None):
        assert not metrics or y_true is not None, (
            "If metrics is true then you should provide y_true"
        )
        results = self.__executed_graph.execute(X, y, best_path=False)
        return results
