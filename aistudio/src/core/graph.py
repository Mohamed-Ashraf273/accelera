import graph


class Graph:
    def __init__(self):
        self._impl = graph.Graph()
        self._nodes = []

    def __call__(self, *args):
        print(f"Graph called with {len(args)} arguments")
        result = self._impl.compute(*args)
        print(f"Graph result: {result}, type: {type(result)}")
        return result

    def add(self, name, py_op=None, *args):
        print(f"Adding node: {name}, py_op: {py_op}, args: {args}")

        if py_op is None:
            node = self._impl.add(name)
            self._nodes.append(node)
        else:
            node = self._impl.add(name, py_op, *args)
            self._nodes.append(node)

            if len(self._nodes) > 1:
                previous_node = self._nodes[-2]
                print(f"  Connecting {previous_node.name} -> {node.name}")
                self._impl.connect(previous_node, node)

        return py_op
