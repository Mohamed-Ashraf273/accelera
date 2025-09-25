class ExecutedGraphWrapper:
    def __init__(self, executed_graph):
        self.__executed_graph = executed_graph
        self.__executed_graph.enableParallelExecution(True)

    def __call__(self, X, metrics=False, y_true=None):
        if metrics:
            if y_true is None:
                raise ValueError("y_true must be provided when metrics=True")

            self.__executed_graph.enableMetrics(y_true)
        results = self.__executed_graph.execute(X, None, best_path=False)
        return results
