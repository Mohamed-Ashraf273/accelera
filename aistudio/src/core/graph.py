import graph

class Graph:
    def __init__(self):
        self._impl = graph.Graph()

    def add(self, name, py_op=None, *args):
        def node(obj, *args):
            if callable(obj):
                return lambda: obj(*args)
            else:
                return lambda: obj
            
        if py_op is None:
            self._impl.add(name)
        else:
            self._impl.add(name, node(py_op, *args))
        return self

    def compute(self):
        return self._impl.compute()
