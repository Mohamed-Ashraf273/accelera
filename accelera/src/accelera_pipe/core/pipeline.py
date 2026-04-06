from accelera.src.accelera_pipe.core.executed_graph import ExecutedGraph
from accelera.src.accelera_pipe.core.node import Node
from accelera.src.accelera_pipe.core.pipeline_base import PipelineBase
from accelera.src.utils.accelera_utils import execute_fit
from accelera.src.utils.accelera_utils import get_correct_metric_class
from accelera.src.utils.accelera_utils import get_metric_object


class Pipeline(PipelineBase):
    def __init__(self):
        super().__init__(_graph=None)

    def __call__(
        self, X, y=None, select_strategy: str = "all", custom_strategy=None
    ):
        results = self._PipelineBase__graph.execute(
            X,
            y=y,
            select_strategy=select_strategy,
            custom_strategy=custom_strategy,
        )
        executed_graph = ExecutedGraph(results[0])
        predictions = results[1:]
        return predictions, executed_graph

    def preprocess(self, name, func, branch=False):
        func_params = {"func": func, "execute_fit": execute_fit}
        if branch:
            return Node("preprocess", name, func_params)

        self._PipelineBase__graph.add_node(
            self.types["preprocess"], name, func_params
        )
        return self

    def model(self, name, model, branch=False):
        model_params = {
            "model": model,
            "execute_fit": execute_fit,
        }
        if branch:
            return Node("model", name, model_params)

        self._PipelineBase__graph.add_node(
            self.types["model"], name, model_params
        )
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
            return Node("predict", name, predict_params)

        self._PipelineBase__graph.add_node(
            self.types["predict"], name, predict_params
        )
        return self

    def metric(
        self,
        name,
        metric_name,
        y_true=None,
        plot_func=None,
        branch=False,
        labels_name=None,
        headers_name=None,
        **params,
    ):
        metric_func = get_metric_object(metric_name)

        if metric_func is not None:
            metric_obj = get_correct_metric_class(
                name,
                metric_func,
                y_true,
                plot_func,
                labels_name,
                headers_name,
                **params,
            )
            if metric_obj is None:
                raise ValueError(
                    f"Metric '{metric_name}' is incompatible with "
                    "the supervised or unsupervised metric structure."
                )

            if branch:
                return Node("metric", name, metric_obj)

            self._PipelineBase__graph.add_node(
                self.types["metric"], name, metric_obj
            )
            return self
        else:
            raise ValueError(f"Metric '{metric_name}' is not recognized.")

    def merge(self, name, strategy="hard_voting", branch=False):
        if branch:
            return Node("merge", name, strategy)

        self._PipelineBase__graph.add_node(self.types["merge"], name, strategy)
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
                if isinstance(node, Node):
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
                        " Node instances. Use the 'branch=True' "
                        "argument when adding nodes. "
                        f"Got: {type(node).__name__}"
                    )

            branches_to_send.append(branch_objects)

        self._PipelineBase__graph.split(
            name, branches_to_send, node_types, node_names
        )
        return self
