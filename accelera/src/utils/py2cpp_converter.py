import ast


class _CppEmitError(ValueError):
    pass


class _PythonToCpp(ast.NodeVisitor):
    def __init__(self):
        self._lines: list[str] = []
        self._indent = 0
        self._declared: set[str] = set()
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
            op = self._binop(node.op)
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
            ast.Pow: None,
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
        self._declared = set()
        for stmt in main_body:
            self.visit(stmt)
        self.emit("return 0;")
        self._declared = prev_declared
        self._indent -= 1
        self.emit("}")

    def visit_Expr(self, node: ast.Expr) -> None:
        self.emit(f"{self.expr(node.value)};")

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise _CppEmitError("multi-target assignment is not supported")
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise _CppEmitError("only simple name assignment is supported")
        name = target.id
        value = self.expr(node.value)
        if name not in self._declared:
            ctype = self._infer_ctype(node.value)
            self._declared.add(name)
            self.emit(f"{ctype} {name} = {value};")
        else:
            self.emit(f"{name} = {value};")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise _CppEmitError("only simple name aug-assign is supported")
        name = node.target.id
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
        cmp = "<" if step is None or not step.strip().startswith("-") else ">"
        step_expr = step if step is not None else "1"
        self.emit(
            f"for (int {var} = {start}; {var} {cmp} {stop}; {var} += {step_expr}) {{"
        )
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
        args = []
        for a in node.args.args:
            args.append(f"auto {a.arg}")
        self.emit(f"auto {node.name}({', '.join(args)}) {{")
        self._indent += 1
        prev_declared = self._declared
        self._declared = set(a.arg for a in node.args.args)
        for s in node.body:
            self.visit(s)
        self._declared = prev_declared
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
                return "int"
            if isinstance(node.value, float):
                return "double"
            if isinstance(node.value, str):
                self._includes.add("#include <string>")
                return "std::string"
        return "auto"

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

    converter = _PythonToCpp()
    try:
        converter.visit(tree)
    except _CppEmitError as e:
        raise ValueError(str(e)) from e
    return converter.render()
