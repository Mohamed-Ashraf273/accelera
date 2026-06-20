from accelera.src.parallelizer.py2cpp_converter import py2cpp_converter


class TestPy2CppConverter:
    def test_converts_print(self):
        cpp = py2cpp_converter('print("hi")')
        assert "#include <iostream>" in cpp
        assert "int main() {" in cpp
        assert "std::cout" in cpp
        assert '"hi"' in cpp
        assert "std::endl" in cpp

    def test_converts_assignment_and_augassign(self):
        cpp = py2cpp_converter("x = 1\nx += 1\ny = 1\ny += 2")
        assert "int main() {" in cpp
        assert "int x = 1;" in cpp
        assert "x++;" in cpp
        assert "y += 2;" in cpp

    def test_converts_power_operator(self):
        cpp = py2cpp_converter("norm = s**0.5")
        assert "#include <cmath>" in cpp
        assert "double norm = std::pow(s, 0.5);" in cpp

    def test_converts_subscript_assignment(self):
        cpp = py2cpp_converter("X[i][j] = X[i][j] / norm")
        assert "X[i][j] = (X[i][j] / norm);" in cpp

    def test_converts_for_range(self):
        cpp = py2cpp_converter("for i in range(0, 3):\n    print(i)")
        assert "int main() {" in cpp
        assert "for (int i = 0; i < 3; i++)" in cpp
        assert "std::cout" in cpp

    def test_converts_function_def(self):
        cpp = py2cpp_converter(
            """
def add(a, b):
    c = a + b
    return c
""".strip()
        )
        assert "template <typename T0, typename T1>" in cpp
        assert "auto add(T0 a, T1 b)" in cpp
        assert "auto c = (a + b);" in cpp
        assert "return c;" in cpp
        assert "int main() {" in cpp
        assert cpp.index("auto add(T0 a, T1 b)") < cpp.index("int main() {")
