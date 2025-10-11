from mainera.src.utils.mainera_utils import get_correct_metric_class
from mainera.src.utils.mainera_utils import get_metric_object
from mainera.src.wrappers.executed_graph_wrapper import ExecutedGraphWrapper
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

    def __call__(
        self, X, y=None, select_strategy: str = "all", custom_strategy=None
    ):
        results = self.__graph.execute(
            X,
            y=y,
            select_strategy=select_strategy,
            custom_strategy=custom_strategy,
        )
        executed_graph = ExecutedGraphWrapper(results[0])
        predictions = results[1:]
        return predictions, executed_graph

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

    def predict(
        self,
        name,
        test_data,
        output_func="predict",
        positive_class=-1,
        branch=False,
    ):
        predict_params = {
            "test_data": test_data,
            "output_func": output_func,
            "positive_class": positive_class,
        }
        if branch:
            return NodeWrapper("predict", name, predict_params)

        self.__graph.add_node(graph.NodeType.PREDICT, name, predict_params)
        return self

    def metric(
        self,
        name,
        metric_name,
        y_true=None,
        tuple_argums=None,
        branch=False,
        **params,
    ):
        metric_func = get_metric_object(metric_name)

        if metric_func is not None:
            metric_obj = get_correct_metric_class(
                name, metric_func, y_true, tuple_argums, **params
            )
            if metric_obj is None:
                raise ValueError(
                    f"Metric '{metric_name}' is incompatible with "
                    "the supervised or unsupervised metric structure."
                )

            if branch:
                return NodeWrapper("metric", name, metric_obj)

            self.__graph.add_node(graph.NodeType.METRIC, name, metric_obj)
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
                    if node.node_type == "merge":
                        raise ValueError(
                            "Branching with Merge nodes is not "
                            "implemented yet. Please add merge nodes "
                            "outside of branches."
                        )
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

    def save_preprocessed_data(self, directory):
        if not self.__graph.savePreprocessedData(directory):
            raise ValueError("Saving data to disk failed.")
        return self
