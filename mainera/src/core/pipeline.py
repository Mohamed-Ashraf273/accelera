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
        self.__graph = graph.Graph()
        self.__graph.enableParallelExecution(True)

    def __call__(self, X, y=None):
        return self.__graph.execute(X, y)

    def preprocess(self, name, func, branch=False):
        if branch:
            return NodeWrapper("preprocess", name, func)

        self.__graph.add_node(graph.NodeType.PREPROCESS, name, func)
        return self

    def model(self, name, model, branch=False):
        if branch:
            return NodeWrapper("model", name, model)

        self.__graph.add_node(graph.NodeType.MODEL, name, model)
        return self

    def predict(self, name, test_data, predict_proba=False, branch=False):
        predict_params = {
            "test_data": test_data,
            "predict_proba": predict_proba,
        }
        if branch:
            return NodeWrapper("predict", name, predict_params)

        self.__graph.add_node(graph.NodeType.PREDICT, name, predict_params)
        return self

    def metric(self, name, metric_func, y_true, branch=False):
        metric_params = {"func": metric_func, "y_true": y_true}
        if branch:
            return NodeWrapper("metric", name, metric_params)

        self.__graph.add_node(graph.NodeType.METRIC, name, metric_params)
        return self

    def branch(self, name, *branches):
        branches_to_send = []
        node_types = []
        node_names = []

        for branch in branches:
            branch_objects = []

            if isinstance(branch, (list, tuple)):
                branch_iter = branch
            elif hasattr(branch, "__iter__") and not isinstance(branch, str):
                branch_iter = branch
            else:
                branch_iter = [branch]

            for node in branch_iter:
                if isinstance(node, NodeWrapper):
                    branch_objects.append(node.obj)
                    node_types.append(node.node_type.upper())
                    node_names.append(node.name)
                else:
                    raise ValueError(
                        "All arguments to branch() must be"
                        " NodeWrapper instances. Use the 'branch=True' "
                        "argument when adding nodes. "
                        f"Got: {type(node).__name__}"
                    )

            branches_to_send.append(branch_objects)

        self.__graph.split(name, branches_to_send, node_types, node_names)
        return self

    def merge(self, name, merge_func):
        raise NotImplementedError("Merge functionality not yet implemented")
        self.__graph.mergeBranches(name, merge_func)
        return self

    def set_multicore_threshold(self, threshold):
        self.__graph.setMulticoreThreshold(threshold)
        return self

    def disable_parallel_execution(self):
        self.__graph.enableParallelExecution(False)
        return self

    def serialize(self, filepath):
        graph.serialize_graph(self.__graph, filepath)
        return self
