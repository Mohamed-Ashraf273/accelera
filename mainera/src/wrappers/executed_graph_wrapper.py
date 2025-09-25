class ExecutedGraphWrapper:
    def __init__(self, executed_graph):
        self.__executed_graph = executed_graph
        self.__executed_graph.enableParallelExecution(True)

    def __call__(self, X, metrics=False, y_true=None):
        if metrics:
            if y_true is None:
                raise ValueError("y_true must be provided when metrics=True")

            self.__executed_graph.enableDisableMetrics(
                y_true=y_true, enable=True
            )
        results = self.__executed_graph.execute(X, best_path=False)
        if metrics:
            self.__executed_graph.enableDisableMetrics(enable=False)
        return results
