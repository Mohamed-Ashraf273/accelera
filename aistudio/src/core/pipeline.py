try:
    import graph
except ImportError as e:
    raise ImportError(
        "The 'graph' C++ module could not be imported. "
        "Please ensure it is built and available in your PYTHONPATH."
    ) from e


class NodeWrapper:
    def __init__(self, node_type, name, obj):
        self.node_type = node_type
        self.name = name
        self.obj = obj


class Pipeline:
    def __init__(self):
        self._graph = graph.Graph()

    def __call__(self, X, y=None):
        return self._graph.execute(X, y)

    def preprocess(self, name, func, branch=False):
        if branch:
            return NodeWrapper("preprocess", name, func)

        self._graph.add_node(graph.NodeType.PREPROCESS, name, func)
        return self

    def model(self, name, model, branch=False):
        if branch:
            return NodeWrapper("model", name, model)

        self._graph.add_node(graph.NodeType.MODEL, name, model)
        return self

    def predict(self, name, test_data, branch=False):
        if branch:
            return NodeWrapper("predict", name, test_data)

        self._graph.add_node(graph.NodeType.PREDICT, name, test_data)
        return self

    def branch(self, name, *branch_nodes):
        branch_objects = []
        node_types = []
        node_names = []

        for node in branch_nodes:
            if isinstance(node, NodeWrapper):
                branch_objects.append(node.obj)
                node_types.append(
                    node.node_type.upper()
                )  # Convert to uppercase
                node_names.append(node.name)
            else:
                branch_objects.append(node)
                node_types.append("MODEL")
                node_names.append(f"auto_{len(branch_objects)}")

        self._graph.split(name, branch_objects, node_types, node_names)
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

    def serialize(self, filepath):
        graph.serialize_graph(self._graph, filepath)
        return self
