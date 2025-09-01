try:
    import graph
except ImportError as e:
    raise ImportError(
        "The 'graph' C++ module could not be imported. "
        "Please ensure it is built and available in your PYTHONPATH."
    ) from e


class Pipeline:
    def __init__(self):
        self._impl = graph.Graph()
        self._nodes = []

    def __call__(self, *args):
        result = self._impl.compute(*args)
        return result

    def add(self, name, py_op=None, *args):
        if py_op is None:
            node = self._impl.add(name)
            self._nodes.append(node)
        else:
            node = self._impl.add(name, py_op, *args)
            self._nodes.append(node)

            if len(self._nodes) > 1:
                previous_node = self._nodes[-2]
                self._impl.connect(previous_node, node)
        if hasattr(py_op, "fit") and callable(getattr(py_op, "fit")):
            return py_op
        return node
