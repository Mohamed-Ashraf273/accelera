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

    def compilation_source(self):
        tree = ast.parse(self.source)
        top_level_functions = [
            node for node in tree.body if isinstance(node, ast.FunctionDef)
        ]
        outer_function = next(
            (node for node in top_level_functions if node.name == self.name),
            None,
        )
        if outer_function is None:
            return self.source

        top_level_helpers = [
            node for node in top_level_functions if node is not outer_function
        ]

        helpers = []

        def collect_helpers(function):
            retained_body = []
            for statement in function.body:
                if isinstance(statement, ast.FunctionDef):
                    collect_helpers(statement)
                    helpers.append(statement)
                else:
                    retained_body.append(statement)
            function.body = retained_body

        collect_helpers(outer_function)
        if not helpers:
            return self.source

        helper_names = {helper.name for helper in helpers}
        available_function_names = helper_names | {
            helper.name for helper in top_level_helpers
        }
        if len(available_function_names) != len(helpers) + len(top_level_helpers):
            raise ValueError("Nested helper function names must be unique.")

        dependencies = {}
        for helper in helpers:
            local_names = {argument.arg for argument in helper.args.args}
            local_names.update(
                node.id
                for node in ast.walk(helper)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            )
            referenced_names = {
                node.id
                for node in ast.walk(helper)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            captured_names = (
                referenced_names
                - local_names
                - available_function_names
                - {
                    "len",
                    "print",
                    "range",
                }
            )
            if captured_names:
                captured = ", ".join(sorted(captured_names))
                raise ValueError(
                    f"Nested helper '{helper.name}' captures unsupported "
                    f"outer variables: {captured}."
                )

            dependencies[helper.name] = {
                node.func.id
                for node in ast.walk(helper)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in helper_names
            }

        helpers_by_name = {helper.name: helper for helper in helpers}
        ordered_helpers = []
        visiting = set()
        visited = set()

        def visit_helper(name):
            if name in visited:
                return
            if name in visiting:
                raise ValueError(
                    "Recursive nested helper functions are not supported."
                )
            visiting.add(name)
            for dependency in dependencies[name]:
                visit_helper(dependency)
            visiting.remove(name)
            visited.add(name)
            ordered_helpers.append(helpers_by_name[name])

        for helper in helpers:
            visit_helper(helper.name)

        compiled_tree = ast.Module(
            body=[*top_level_helpers, *ordered_helpers, outer_function],
            type_ignores=[],
        )
        ast.fix_missing_locations(compiled_tree)
        return ast.unparse(compiled_tree)

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

        outer_function = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef)),
            None,
        )
        if outer_function is not None:
            helper_nodes = []
            visited_functions = set()

            def collect_helper(helper):
                if helper is func or id(helper) in visited_functions:
                    return
                if getattr(helper, "__closure__", None):
                    raise ValueError(
                        "Cannot save helper functions with closure variables. "
                        "Pass required values as explicit parameters."
                    )

                visited_functions.add(id(helper))
                helper_source = textwrap.dedent(inspect.getsource(helper))
                helper_tree = ast.parse(helper_source)
                helper_node = next(
                    (
                        node
                        for node in helper_tree.body
                        if isinstance(node, ast.FunctionDef)
                    ),
                    None,
                )
                if helper_node is None:
                    raise ValueError(
                        f"Cannot save helper function '{helper.__name__}' source."
                    )

                referenced_names = {
                    node.id
                    for node in ast.walk(helper_node)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                }
                for referenced_name in referenced_names:
                    candidate = helper.__globals__.get(referenced_name)
                    if inspect.isfunction(candidate):
                        collect_helper(candidate)
                helper_nodes.append(helper_node)

            referenced_names = {
                node.id
                for node in ast.walk(outer_function)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            for referenced_name in referenced_names:
                candidate = func.__globals__.get(referenced_name)
                if inspect.isfunction(candidate):
                    collect_helper(candidate)

            helper_names = {node.name for node in helper_nodes}
            if len(helper_names) != len(helper_nodes):
                raise ValueError("Referenced helper function names must be unique.")

            bundled_tree = ast.Module(
                body=[*helper_nodes, outer_function],
                type_ignores=[],
            )
            ast.fix_missing_locations(bundled_tree)
            return outer_function.name, ast.unparse(bundled_tree)

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
