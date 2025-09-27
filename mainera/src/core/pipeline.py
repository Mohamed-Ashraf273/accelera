from mainera.src.utils.mainera_utils import get_learning_type
from mainera.src.utils.mainera_utils import get_metric_object
from mainera.src.utils.mainera_utils import metric_validation
from mainera.src.wrappers.executed_graph_wrapper import ExecutedGraphWrapper
from mainera.src.wrappers.metric_wrapper import MetricWrapper
from mainera.src.wrappers.node_wrapper import NodeWrapper

try:
    import graph
except ImportError as e:
    raise ImportError(
        "The 'graph' C++ module could not be imported. "
        "Please ensure it is built and available in your PYTHONPATH."
    ) from e


class Pipeline:
    def __init__(self):
        self.__graph = graph.Graph()
        self.__graph.enableParallelExecution(True)

    def __call__(self, X, y=None, best_path=False):
        results = self.__graph.execute(X, y, best_path=best_path)
        executed_graph = ExecutedGraphWrapper(results[0])
        predictions = results[1:]
        return predictions, executed_graph

    def preprocess(self, name, func, branch=False):
        if branch:
            return NodeWrapper("preprocess", name, func)

        self.__graph.add_node(graph.NodeType.PREPROCESS, name, func)
        return self

    def model(self, name, model, branch=False):
        model_type = get_learning_type(model)
        model_params = {"model": model, "model_type": model_type}
        if branch:
            return NodeWrapper("model", name, model_params)

        self.__graph.add_node(graph.NodeType.MODEL, name, model_params)
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

    def metric(
        self,
        name,
        metric_name,
        y_true,
        binary_proba=False,
        branch=False,
        **params,
    ):
        metric_func = get_metric_object(metric_name)

        if metric_func is not None:
            metric_validation(metric_func, metric_name)
            metric_obj = MetricWrapper(metric_func, binary_proba, **params)
            metric_params = {
                "func": metric_obj,
                "metric_name": metric_name,
                "y_true": y_true,
            }

            if branch:
                return NodeWrapper("metric", name, metric_params)

            self.__graph.add_node(graph.NodeType.METRIC, name, metric_params)
            return self
        else:
            raise ValueError(f"Metric '{metric_name}' is not recognized.")

    def merge(self, name, strategy="hard_voting", branch=False):
        if branch:
            return NodeWrapper("merge", name, strategy)

        self.__graph.add_node(graph.NodeType.MERGE, name, strategy)
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

    def set_multicore_threshold(self, threshold):
        self.__graph.setMulticoreThreshold(threshold)
        return self

    def disable_parallel_execution(self):
        self.__graph.enableParallelExecution(False)
        return self

    def serialize(self, filepath):
        graph.serialize_graph(self.__graph, filepath)
        return self
