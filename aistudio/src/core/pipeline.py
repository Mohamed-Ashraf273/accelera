try:
    import graph
except ImportError as e:
    raise ImportError(
        "The 'graph' C++ module could not be imported. "
        "Please ensure it is built and available in your PYTHONPATH."
    ) from e


class Pipeline:
    def __init__(self):
        self._graph = graph.Graph()

    def __call__(self, X, y=None):
        """Execute the pipeline - all execution logic is in C++"""
        return self._graph.execute(X, y)

    def preprocess(self, name, func):
        """Add a preprocessing node - all logic in C++"""
        if self._graph.isBranched():
            self._graph.addToBranches("preprocess", name, func)
        else:
            self._graph.addSequentialNode("preprocess", name, func)
        return self

    def model(self, name, model):
        """Add a model node - all logic in C++"""
        if self._graph.isBranched():
            self._graph.addToBranches("model", name, model)
        else:
            self._graph.addSequentialNode("model", name, model)
        return self

    def predict(self, name, test_data):
        """Add a prediction node - all logic in C++"""
        if self._graph.isBranched():
            self._graph.addToBranches("predict", name, test_data)
        else:
            self._graph.addSequentialNode("predict", name, test_data)
        return self

    def branch(self, name, *branch_models):
        """Create branches - all logic in C++"""
        self._graph.startBranching(name, list(branch_models))
        return self

    def merge(self, name, merge_func):
        """Merge branches - all logic in C++"""
        self._graph.mergeBranches(name, merge_func)
        return self

    # Legacy compatibility methods
    def compile(self):
        return self

    def serialize(self, filepath):
        self._graph.serialize(filepath)
        return self
