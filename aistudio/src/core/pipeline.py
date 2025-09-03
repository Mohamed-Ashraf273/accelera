try:
    import graph
except ImportError as e:
    raise ImportError(
        "The 'graph' C++ module could not be imported. "
        "Please ensure it is built and available in your PYTHONPATH."
    ) from e


class NodeWrapper:
    """Wrapper for nodes when used in branches"""

    def __init__(self, node_type, name, obj):
        self.node_type = node_type  # "preprocess", "model", "predict"
        self.name = name
        self.obj = obj


class Pipeline:
    def __init__(self):
        self._graph = graph.Graph()

    def __call__(self, X, y=None):
        """Execute the pipeline - all execution logic is in C++"""
        return self._graph.execute(X, y)

    def preprocess(self, name, func, branch=False):
        """Add a preprocessing node - all logic in C++"""
        if branch:
            return NodeWrapper("preprocess", name, func)

        if self._graph.isBranched():
            self._graph.addToBranches("preprocess", name, func)
        else:
            self._graph.addSequentialNode("preprocess", name, func)
        return self

    def model(self, name, model, branch=False):
        """Add a model node - all logic in C++"""
        if branch:
            return NodeWrapper("model", name, model)

        if self._graph.isBranched():
            self._graph.addToBranches("model", name, model)
        else:
            self._graph.addSequentialNode("model", name, model)
        return self

    def predict(self, name, test_data, branch=False):
        """Add a prediction node - all logic in C++"""
        if branch:
            return NodeWrapper("predict", name, test_data)

        if self._graph.isBranched():
            self._graph.addToBranches("predict", name, test_data)
        else:
            self._graph.addSequentialNode("predict", name, test_data)
        return self

    def branch(self, name, *branch_nodes):
        """Create branches with any node types - all logic in C++"""
        # Extract objects and types from NodeWrapper instances
        branch_objects = []
        node_types = []
        node_names = []

        for node in branch_nodes:
            if isinstance(node, NodeWrapper):
                branch_objects.append(node.obj)
                node_types.append(node.node_type)
                node_names.append(node.name)
            else:
                # Legacy support - assume it's a model if not wrapped
                branch_objects.append(node)
                node_types.append("model")
                node_names.append(f"auto_{len(branch_objects)}")

        # Pass type information to C++ backend
        self._graph.startBranchingWithTypes(
            name, branch_objects, node_types, node_names
        )
        return self

    def merge(self, name, merge_func):
        """Merge branches - all logic in C++"""
        self._graph.mergeBranches(name, merge_func)
        return self

    def enable_parallel(self, enable=True):
        """Enable or disable parallel execution"""
        self._graph.enableParallelExecution(enable)
        return self

    def set_multicore_threshold(self, threshold):
        """Set minimum number of tasks to use multicore execution"""
        self._graph.setMulticoreThreshold(threshold)
        return self

    # Legacy compatibility methods
    def compile(self):
        return self

    def serialize(self, filepath):
        self._graph.serialize(filepath)
        return self
