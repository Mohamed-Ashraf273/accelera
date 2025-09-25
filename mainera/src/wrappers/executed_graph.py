class ExecutedGraphWrapper:
    def __init__(self, executed_graph):
        self.__executed_graph = executed_graph
        self.__executed_graph.enableParallelExecution(True)

    def __call__(self, X, y):
        results = self.__executed_graph.execute(X, y, best_path=False)
        print(len(results))
        return results
