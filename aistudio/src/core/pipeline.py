import graph


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

        return py_op
