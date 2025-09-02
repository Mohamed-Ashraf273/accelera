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
        self._graph.enableParallelExecution(True)  # Enable parallel execution
        self._first_node = None
        self._last_node = None
        self._compiled = False
        self._branch_paths = []  # Track active branch paths
        self._node_registry = {}  # Store nodes by name

    def __call__(self, X, y=None):
        predictions = self._graph.execute(X, y)
        return predictions

    def preprocess(self, name, preprocessor):
        node = self._graph.add_node(
            graph.NodeType.PREPROCESS, name, preprocessor
        )
        self._node_registry[name] = node
        self._add_node(node)
        return self

    def feature(self, name, feature_extractor):
        node = self._graph.add_node(
            graph.NodeType.FEATURE, name, feature_extractor
        )
        self._node_registry[name] = node
        self._add_node(node)
        return self

    def model(self, name, model):
        node = self._graph.add_node(graph.NodeType.MODEL, name, model, 2, 1)
        self._node_registry[name] = node
        self._add_node(node)
        return self  # Return pipeline for chaining

    def predict(self, name, test_data=None):
        if not self._branch_paths:
            # Normal mode - create single predict node
            node = self._graph.add_node(
                graph.NodeType.PREDICT, name, lambda x: x, 2, 1
            )
            if test_data is not None:
                self._graph.set_input(node, 0, test_data)
            self._node_registry[name] = node
            self._add_node(node)
        else:
            predict_nodes = self._graph.createPredictForBranches(
                name, self._branch_paths, test_data
            )

            # Store predict nodes and clear branch state
            for i, node in enumerate(predict_nodes):
                self._node_registry[f"{name}_branch{i + 1}"] = node
            self._predict_nodes = predict_nodes
            self._branch_paths = []  # Clear branch state

        return self

    def branch(self, name, *branch_objects, merge_func=None):
        """Create a branch node that splits data to multiple models.

        Args:
            name: Branch node name
            *branch_objects: Node objects or node names to branch to
            merge_func: Optional merge function (not implemented yet)
        """
        # Convert node objects to actual nodes,
        # collect node names to get actual nodes
        branch_nodes = []
        for obj in branch_objects:
            if hasattr(obj, "name"):
                branch_nodes.append(obj)
            elif isinstance(obj, str):
                node = self._node_registry.get(obj, None)
                if node:
                    branch_nodes.append(node)
                else:
                    raise ValueError(f"Node '{obj}' not found")
            else:
                raise ValueError(f"Invalid branch object: {obj}")

        branch_node = self._graph.createBranch(name, branch_nodes, merge_func)
        self._node_registry[name] = branch_node

        self._branch_paths = branch_nodes
        self._last_node = None

        return self

    def _add_node(self, node):
        if self._branch_paths:
            # Branch mode - logic handled in predict() method
            return self
        else:
            # Normal single-path mode - simple connection
            if (
                self._last_node
                and hasattr(node, "type")
                and node.type == graph.NodeType.PREDICT
            ):
                # Connect model to predict node
                # (model output -> predict input 1)
                self._last_node.connectTo(0, node, 1)
            elif self._last_node:
                # Normal connection (output -> input)
                self._last_node.connectTo(0, node, 0)
            else:
                # First node
                self._first_node = node
            self._last_node = node
            return self

    def compile(self):
        self._graph.compile()
        self._compiled = True
        return self

    def serialize(self, filepath):
        """Serialize the pipeline graph to an XML file for visualization.

        Args:
            filepath (str): Path to save the XML file

        Returns:
            self: For method chaining
        """
        self._graph.serialize(filepath)
        return self
