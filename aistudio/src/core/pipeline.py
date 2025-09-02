try:
    import pipeline
except ImportError as e:
    raise ImportError(
        "The 'pipeline' C++ module could not be imported. "
        "Please ensure it is built and available in your PYTHONPATH."
    ) from e


class Pipeline:
    def __init__(self):
        self._impl = pipeline.Pipeline()
        self._nodes = []

    def __call__(self, *args):
        return self._impl.compute(*args)

    def add(self, name=None, py_op=None, *args):
        if name is None:
            name = f"node_{len(self._nodes)}"

        if py_op is None:
            raise ValueError("py_op must be provided")

        node = self._impl.add(name, py_op, *args)
        self._nodes.append(node)

        if hasattr(py_op, "fit") and callable(getattr(py_op, "fit")):
            return py_op

        return node
