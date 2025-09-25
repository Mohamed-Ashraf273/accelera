class ExecutedGraphWrapper:
    def __init__(self, executed_graph):
        self.__executed_graph = executed_graph

    def __call__(self):
        return self.__executed_graph
