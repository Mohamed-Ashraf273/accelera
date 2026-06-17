import ast
import inspect
import textwrap

import numpy as np


class SourceBackedFunction:
    def __init__(self, func, runtime_func=None):
        self.name, self.source = self._extract_source(func)
        self._func = runtime_func or func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __getstate__(self):
        return {
            "name": self.name,
            "source": self.source,
        }

    def __setstate__(self, state):
        self.name = state["name"]
        self.source = state["source"]
        self._func = self._restore_function(self.name, self.source)

    def set_runtime_func(self, runtime_func):
        self._func = runtime_func
        return self

    @staticmethod
    def _extract_source(func):
        if getattr(func, "__closure__", None):
            raise ValueError(
                "Cannot save custom preprocess functions with closure "
                "variables. Use a top-level def or a lambda without captured "
                "external variables."
            )

        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name, ast.unparse(node)

        lambda_node = next(
            (node for node in ast.walk(tree) if isinstance(node, ast.Lambda)),
            None,
        )
        if lambda_node is not None:
            name = "_accelera_saved_lambda"
            return name, f"{name} = {ast.unparse(lambda_node)}"

        raise ValueError(
            "Cannot save custom preprocess function source. Use a normal "
            "def function or a simple lambda."
        )

    @staticmethod
    def _restore_function(name, source):
        namespace = {"np": np, "numpy": np}
        exec(source, namespace)
        return namespace[name]
