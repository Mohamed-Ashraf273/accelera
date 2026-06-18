import ast


class _CppEmitError(ValueError):
    pass


class _WideIntAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.wide_int_vars: set[str] = set()
        self._loop_depth = 0

    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._loop_depth > 0 and isinstance(node.target, ast.Name):
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                self.wide_int_vars.add(node.target.id)
        self.generic_visit(node)


class _PythonToCpp(ast.NodeVisitor):
    def __init__(self, wide_int_vars: set[str] | None = None):
        self._lines: list[str] = []
        self._indent = 0
        self._declared: set[str] = set()
        self._var_types: dict[str, str] = {}
        self._wide_int_vars = wide_int_vars or set()
        self._includes: set[str] = {"#include <iostream>"}

    def emit(self, line: str = "") -> None:
        self._lines.append(("    " * self._indent) + line)

    def render(self) -> str:
        includes = sorted(self._includes)
        return (
            "\n".join(includes + ([""] if includes else []) + self._lines).rstrip()
            + "\n"
        )

    def expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "nullptr"
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            if isinstance(node.value, str):
                return (
                    '"' + node.value.replace("\\", "\\\\").replace('"', '\\"') + '"'
                )
            return repr(node.value)

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                self._includes.add("#include <cmath>")
                return f"std::pow({self.expr(node.left)}, {self.expr(node.right)})"

            op = self._binop(node.op)

            if isinstance(node.op, ast.Mult):
                result_type = self._infer_ctype(node)
                if result_type == "long long":
                    return (
                        f"(static_cast<long long>({self.expr(node.left)}) "
                        f"* {self.expr(node.right)})"
                    )

            return f"({self.expr(node.left)} {op} {self.expr(node.right)})"

        if isinstance(node, ast.UnaryOp):
            op = self._unaryop(node.op)
            return f"({op}{self.expr(node.operand)})"

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise _CppEmitError("chained comparisons are not supported")
            op = self._cmpop(node.ops[0])
            return f"({self.expr(node.left)} {op} {self.expr(node.comparators[0])})"

        if isinstance(node, ast.BoolOp):
            op = "&&" if isinstance(node.op, ast.And) else "||"
            parts = [self.expr(v) for v in node.values]
            return "(" + f" {op} ".join(parts) + ")"

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                return self._emit_print_expr(node)

            func = self.expr(node.func)
            args = ", ".join(self.expr(a) for a in node.args)
            return f"{func}({args})"

        if isinstance(node, ast.Attribute):
            return f"{self.expr(node.value)}.{node.attr}"

        if isinstance(node, ast.Subscript):
            return f"{self.expr(node.value)}[{self.expr(node.slice)}]"

        if isinstance(node, ast.Slice):
            raise _CppEmitError("slice syntax is not supported")

        raise _CppEmitError(f"unsupported expression: {type(node).__name__}")

    def _emit_print_expr(self, node: ast.Call) -> str:
        if node.keywords:
            raise _CppEmitError("print() keywords are not supported")
        self._includes.add("#include <iostream>")
        if not node.args:
            return "(std::cout << std::endl)"
        stream = "std::cout"
        for i, arg in enumerate(node.args):
            stream += f" << {self.expr(arg)}"
            if i != len(node.args) - 1:
                stream += ' << " "'
        stream += " << std::endl"
        return f"({stream})"

    def _binop(self, op: ast.operator) -> str:
        table = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
        }
        out = table.get(type(op))
        if out is None:
            raise _CppEmitError(f"unsupported binary operator: {type(op).__name__}")
        return out

    def _unaryop(self, op: ast.unaryop) -> str:
        table = {ast.UAdd: "+", ast.USub: "-", ast.Not: "!"}
        out = table.get(type(op))
        if out is None:
            raise _CppEmitError(f"unsupported unary operator: {type(op).__name__}")
        return out

    def _cmpop(self, op: ast.cmpop) -> str:
        table = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        out = table.get(type(op))
        if out is None:
            raise _CppEmitError(
                f"unsupported comparison operator: {type(op).__name__}"
            )
        return out

    def visit_Module(self, node: ast.Module) -> None:
        function_defs = [
            stmt for stmt in node.body if isinstance(stmt, ast.FunctionDef)
        ]
        main_body = [
            stmt for stmt in node.body if not isinstance(stmt, ast.FunctionDef)
        ]

        for stmt in function_defs:
            self.visit(stmt)

        self.emit("int main() {")
        self._indent += 1
        prev_declared = self._declared
        prev_var_types = self._var_types
        self._declared = set()
        self._var_types = {}
        for stmt in main_body:
            self.visit(stmt)
        self.emit("return 0;")
        self._declared = prev_declared
        self._var_types = prev_var_types
        self._indent -= 1
        self.emit("}")

    def visit_Expr(self, node: ast.Expr) -> None:
        self.emit(f"{self.expr(node.value)};")

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise _CppEmitError("multi-target assignment is not supported")
        target = node.targets[0]
        value = self.expr(node.value)
        if not isinstance(target, ast.Name):
            self.emit(f"{self.expr(target)} = {value};")
            return
        name = target.id
        if name not in self._declared:
            ctype = (
                "long long"
                if name in self._wide_int_vars
                else self._infer_ctype(node.value)
            )
            self._declared.add(name)
            self._var_types[name] = ctype
            self.emit(f"{ctype} {name} = {value};")
        else:
            self._var_types[name] = self._infer_ctype(node.value)
            self.emit(f"{name} = {value};")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise _CppEmitError("only simple name aug-assign is supported")
        name = node.target.id
        if isinstance(node.op, ast.Add) and self._is_constant_one(node.value):
            self.emit(f"{name}++;")
            return
        op = self._binop(node.op)
        self.emit(f"{name} {op}= {self.expr(node.value)};")

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            self.emit("return;")
        else:
            self.emit(f"return {self.expr(node.value)};")

    def visit_If(self, node: ast.If) -> None:
        self.emit(f"if {self.expr(node.test)} {{")
        self._indent += 1
        for s in node.body:
            self.visit(s)
        self._indent -= 1
        if node.orelse:
            self.emit("} else {")
            self._indent += 1
            for s in node.orelse:
                self.visit(s)
            self._indent -= 1
        self.emit("}")

    def visit_For(self, node: ast.For) -> None:
        if not isinstance(node.target, ast.Name):
            raise _CppEmitError("only simple for-loop targets are supported")
        var = node.target.id
        start, stop, step = self._parse_range_iter(node.iter)
        if var not in self._declared:
            self._declared.add(var)
            self._var_types[var] = "int"
        cmp = "<" if step is None or not step.strip().startswith("-") else ">"
        step_expr = step if step is not None else "1"
        increment = f"{var}++" if step_expr == "1" else f"{var} += {step_expr}"
        self.emit(f"for (int {var} = {start}; {var} {cmp} {stop}; {increment}) {{")
        self._indent += 1
        for s in node.body:
            self.visit(s)
        self._indent -= 1
        self.emit("}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.decorator_list:
            raise _CppEmitError("decorators are not supported")
        if node.returns is not None:
            pass
        template_params = [f"typename T{i}" for i, _ in enumerate(node.args.args)]
        args = [f"T{i} {a.arg}" for i, a in enumerate(node.args.args)]
        if template_params:
            self.emit(f"template <{', '.join(template_params)}>")
        self.emit(f"auto {node.name}({', '.join(args)}) {{")
        self._indent += 1
        prev_declared = self._declared
        prev_var_types = self._var_types
        self._declared = set(a.arg for a in node.args.args)
        self._var_types = {a.arg: "auto" for a in node.args.args}
        for s in node.body:
            self.visit(s)
        self._declared = prev_declared
        self._var_types = prev_var_types
        self._indent -= 1
        self.emit("}")
        self.emit("")

    def generic_visit(self, node: ast.AST) -> None:
        raise _CppEmitError(f"unsupported syntax: {type(node).__name__}")

    def _infer_ctype(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
            if isinstance(node.value, int):
                if -(2**31) <= node.value <= 2**31 - 1:
                    return "int"
                return "long long"
            if isinstance(node.value, float):
                return "double"
            if isinstance(node.value, str):
                self._includes.add("#include <string>")
                return "std::string"

        if isinstance(node, ast.Name):
            return self._var_types.get(node.id, "auto")

        if isinstance(node, ast.BinOp):
            left_type = self._infer_ctype(node.left)
            right_type = self._infer_ctype(node.right)

            if left_type == "double" or right_type == "double":
                return "double"

            if isinstance(node.op, ast.Div):
                return "double"

            if isinstance(node.op, ast.Mult):
                if left_type in {"int", "long long"} and right_type in {
                    "int",
                    "long long",
                }:
                    return "long long"

            if left_type == "long long" or right_type == "long long":
                return "long long"

            if left_type == "int" and right_type == "int":
                return "int"

        return "auto"

    def _is_constant_one(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and node.value == 1
        )

    def _parse_range_iter(self, node: ast.AST) -> tuple[str, str, str | None]:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and not node.keywords
        ):
            raise _CppEmitError("for-loop only supports range(...)")
        if not (1 <= len(node.args) <= 3):
            raise _CppEmitError("range() with 1..3 args is supported")

        if len(node.args) == 1:
            return "0", self.expr(node.args[0]), None
        if len(node.args) == 2:
            return self.expr(node.args[0]), self.expr(node.args[1]), None
        return (
            self.expr(node.args[0]),
            self.expr(node.args[1]),
            self.expr(node.args[2]),
        )


def py2cpp_converter(python_code: str) -> str:
    try:
        tree = ast.parse(python_code)
    except SyntaxError as e:
        raise ValueError(f"Invalid python code: {e}") from e

    analyzer = _WideIntAnalyzer()
    analyzer.visit(tree)

    converter = _PythonToCpp(analyzer.wide_int_vars)
    try:
        converter.visit(tree)
    except _CppEmitError as e:
        raise ValueError(str(e)) from e
    return converter.render()
