import json
import os
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from accelera.src.parallelizer.parallelizer import Parallelizer
from accelera.src.parallelizer.parallelizer import _resolve_loop_class
from accelera.src.parallelizer.parallelizer import extract_loops
from accelera.src.parallelizer.parallelizer import write_loops_to_json
from accelera.src.utils.source_backed_function import SourceBackedFunction


def normalize_rows_for_parallelizer_test(X):
    for i in range(len(X)):
        s = 0
        for j in range(len(X[i])):
            s += X[i][j] * X[i][j]

        norm = s**0.5

        for j in range(len(X[i])):
            X[i][j] = X[i][j] / norm

    return X


def square_rows_with_nested_helper_for_parallelizer_test(X):
    def square(value):
        return value * value

    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = square(X[i][j])

    return X


def gamma_correction_with_named_input_for_parallelizer_test(image):
    def adjust_pixel(pixel):
        return (pixel / 255.0) ** 0.8 * 255.0

    for i in range(len(image)):
        for j in range(len(image[i])):
            image[i][j] = adjust_pixel(image[i][j])

    return image


def multiply_for_parallelizer_test(value, factor):
    return value * factor


def gamma_correction_with_external_helper_for_parallelizer_test(image):
    def adjust_pixel(pixel):
        return multiply_for_parallelizer_test(pixel, 2.2)

    for i in range(len(image)):
        for j in range(len(image[i])):
            image[i][j] = adjust_pixel(image[i][j])

    return image


class NormalizeRowsInstanceForParallelizerTest:
    def transform(self, X):
        for i in range(len(X)):
            s = 0
            for j in range(len(X[i])):
                s += X[i][j] * X[i][j]

            norm = s**0.5

            for j in range(len(X[i])):
                X[i][j] = X[i][j] / norm

        return X


class TestParallelizer:
    def test_classify_returns_prediction(self, monkeypatch):
        parallelizer = Parallelizer()

        class DummyResponse:
            status_code = 200

            def json(self):
                return {"result": "parallel_for"}

        request_payload = {}

        def fake_post(url, json, timeout):
            request_payload.update(json)
            return DummyResponse()

        monkeypatch.setattr(
            "accelera.src.parallelizer.parallelizer.requests.post",
            fake_post,
        )

        loop_code = "for (int i = 0; i < n; ++i) { out[i] = in[i]; }"
        result = parallelizer._classify(loop_code)

        assert result == "parallel_for"
        assert "embedding" in request_payload
        assert len(request_payload["embedding"]) == 40

    def test_generate_omp_pragma_with_loop_none_returns_loop(self):
        parallelizer = Parallelizer()
        loop_code = "for (int i = 0; i < n; ++i) {\n    sum += a[i];\n}"

        result = parallelizer._generate_omp_pragma_with_loop(loop_code, "none")

        assert result == loop_code

    def test_resolve_loop_class_upgrades_obvious_reduction(self):
        loop_code = "for (int i = 0; i < 5; i++) {\n    sum += i;\n}"

        result = _resolve_loop_class(loop_code, "none")

        assert result == "reduction"

    def test_resolve_loop_class_upgrades_independent_array_write(self):
        loop_code = (
            "for (int i = 0; i < n; i++) {\n    input[i] = (i % 1000) * 0.001;\n}"
        )

        result = _resolve_loop_class(loop_code, "none")

        assert result == "parallel_for"

    def test_generate_omp_pragma_with_loop_adds_validated_pragma(self, monkeypatch):
        parallelizer = Parallelizer()

        class DummyResponse:
            status_code = 200

            def json(self):
                return {"pragma": "omp parallel for"}

        monkeypatch.setattr(
            "accelera.src.parallelizer.parallelizer.requests.post",
            lambda url, json, timeout: DummyResponse(),
        )
        result = parallelizer._generate_omp_pragma_with_loop(
            "for (int i = 0; i < n; ++i) {}", "parallel_for"
        )

        assert result.startswith("#pragma omp parallel for\n")
        assert "for (int i = 0; i < n; ++i) {}" in result

    def test_generate_omp_pragma_does_not_reduce_local_loop_variable(self):
        parallelizer = Parallelizer()
        loop_code = (
            "for (int i = 0; i < len(X); i++) {\n"
            "    int s = 0;\n"
            "    for (int j = 0; j < len(X[i]); j++) {\n"
            "        s += (X[i][j] * X[i][j]);\n"
            "    }\n"
            "}"
        )

        result = parallelizer._generate_omp_pragma_with_loop(
            loop_code, "parallel_for"
        )

        assert result.startswith("#pragma omp parallel for\n")
        assert "reduction(+ : s)" not in result

    def test_parallelize_writes_parallelized_output(self, monkeypatch, tmp_path):
        source_file = tmp_path / "sample.c"
        source_file.write_text(
            "int main() {\n"
            "for (int i = 0; i < n; ++i) {\n"
            "    sum += a[i];\n"
            "}\n"
            "return 0;\n"
            "}\n"
        )

        loop_code = "for (int i = 0; i < n; ++i) {\n    sum += a[i];\n}"

        parallelizer = Parallelizer()
        parallelizer.cache_dir = tmp_path / "cache"

        monkeypatch.setattr(
            "accelera.src.parallelizer.parallelizer.extract_loops",
            lambda file_path: [SimpleNamespace()],
        )

        def fake_write_loops_to_json(loops, output_json):
            Path(output_json).write_text(
                json.dumps(
                    [
                        {
                            "code": loop_code,
                            "start_line": 2,
                            "end_line": 4,
                            "type": "for",
                        }
                    ]
                )
            )
            return True

        monkeypatch.setattr(
            "accelera.src.parallelizer.parallelizer.write_loops_to_json",
            fake_write_loops_to_json,
        )
        monkeypatch.setattr(
            "accelera.src.parallelizer.parallelizer.extract_features",
            lambda code: {"dummy": True},
        )
        monkeypatch.setattr(parallelizer, "_classify", lambda code: "omp")
        monkeypatch.setattr(
            parallelizer,
            "_generate_omp_pragma_with_loop",
            lambda code, cls, code_context="": f"#pragma omp parallel for\n{code}",
        )

        result = parallelizer.parallelize(str(source_file), output_dir=tmp_path)

        output_file = tmp_path / "parallelized_sample.c"
        assert result is None
        assert output_file.exists()
        assert "#pragma omp parallel for" in output_file.read_text()

    def test_parallelize_code_string_returns_parallelized_code(
        self, monkeypatch, tmp_path
    ):
        code = (
            "int main() {\n"
            "for (int i = 0; i < n; ++i) {\n"
            "    out[i] = in[i];\n"
            "}\n"
            "return 0;\n"
            "}\n"
        )
        loop_code = "for (int i = 0; i < n; ++i) {\n    out[i] = in[i];\n}"

        parallelizer = Parallelizer()
        parallelizer.cache_dir = tmp_path / "cache"
        monkeypatch.setattr(
            "accelera.src.parallelizer.parallelizer.extract_loops",
            lambda code: [
                SimpleNamespace(
                    code=loop_code,
                    start_line=2,
                    end_line=4,
                    type="for",
                )
            ],
        )
        monkeypatch.setattr(parallelizer, "_classify", lambda code: "parallel_for")

        result = parallelizer.parallelize(code)

        assert "#pragma omp parallel for" in result
        assert not list(tmp_path.rglob("parallelized_*.c"))
        assert not list(tmp_path.rglob("extracted_loops_*.json"))

    def test_parallelize_uses_rule_based_fallback_when_classifier_fails(
        self, monkeypatch, tmp_path
    ):
        code = (
            "int main() {\n"
            "for (int i = 0; i < n; ++i) {\n"
            "    for (int j = 0; j < m; ++j) {\n"
            "        out[i][j] = in[i][j];\n"
            "    }\n"
            "}\n"
            "return 0;\n"
            "}\n"
        )
        loop_code = (
            "for (int i = 0; i < n; ++i) {\n"
            "    for (int j = 0; j < m; ++j) {\n"
            "        out[i][j] = in[i][j];\n"
            "    }\n"
            "}"
        )

        parallelizer = Parallelizer()
        parallelizer.cache_dir = tmp_path / "cache"
        monkeypatch.setattr(
            "accelera.src.parallelizer.parallelizer.extract_loops",
            lambda code: [
                SimpleNamespace(
                    code=loop_code,
                    start_line=2,
                    end_line=6,
                    type="for",
                )
            ],
        )
        monkeypatch.setattr(
            parallelizer,
            "_classify",
            lambda code: (_ for _ in ()).throw(RuntimeError("offline")),
        )

        result = parallelizer.parallelize(code)

        assert "#pragma omp parallel for collapse(2)" in result

    def test_optimize_pymethod_falls_back_for_unsupported_function(self):
        parallelizer = Parallelizer()

        def unsupported_tuple_return(x):
            return x, x

        result = parallelizer._optimize_pymethod(unsupported_tuple_return)

        assert result is unsupported_tuple_return

    def test_optimize_pymethod_compiles_supported_numpy_loop(self):
        parallelizer = Parallelizer()
        X = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float64)

        result = parallelizer._optimize_pymethod(
            normalize_rows_for_parallelizer_test
        )
        normalized = result(X.copy())

        assert result is not normalize_rows_for_parallelizer_test
        assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)

    def test_optimize_pymethod_compiles_nested_helper(self):
        parallelizer = Parallelizer()
        X = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)

        result = parallelizer._optimize_pymethod(
            square_rows_with_nested_helper_for_parallelizer_test
        )

        assert result is not square_rows_with_nested_helper_for_parallelizer_test
        assert np.array_equal(result(X.copy()), X**2)

    def test_optimize_pymethod_preserves_input_parameter_name(self):
        parallelizer = Parallelizer()
        image = np.array([[0.0, 128.0], [255.0, 64.0]], dtype=np.float64)

        result = parallelizer._optimize_pymethod(
            gamma_correction_with_named_input_for_parallelizer_test
        )

        assert result is not gamma_correction_with_named_input_for_parallelizer_test
        assert np.allclose(
            result(image.copy()),
            gamma_correction_with_named_input_for_parallelizer_test(image.copy()),
        )

    def test_optimize_pymethod_compiles_referenced_helper(self):
        parallelizer = Parallelizer()
        image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        result = parallelizer._optimize_pymethod(
            gamma_correction_with_external_helper_for_parallelizer_test
        )

        assert (
            result is not gamma_correction_with_external_helper_for_parallelizer_test
        )
        assert np.allclose(
            result(image.copy()),
            gamma_correction_with_external_helper_for_parallelizer_test(
                image.copy()
            ),
        )

    def test_parallelized_nested_helper_is_pickleable(self):
        parallelizer = Parallelizer()
        X = np.array([[2.0, 3.0]], dtype=np.float64)

        result = parallelizer.parallelize(
            square_rows_with_nested_helper_for_parallelizer_test
        )
        restored_result = pickle.loads(pickle.dumps(result))

        assert isinstance(result, SourceBackedFunction)
        assert np.array_equal(result(X.copy()), X**2)
        assert np.array_equal(restored_result(X.copy()), X**2)

    def test_optimize_pyinstance_compiles_transform(self):
        parallelizer = Parallelizer()
        instance = NormalizeRowsInstanceForParallelizerTest()
        X = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float64)

        result = parallelizer._optimize_pyinstance(instance)
        normalized = result.transform(X.copy())

        assert result is instance
        assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)

    def test_parallelize_skips_inner_loop_when_outer_selected(
        self, monkeypatch, tmp_path
    ):
        source_file = tmp_path / "nested.c"
        source_file.write_text(
            "int main() {\n"
            "for (int r = 0; r < rows; r++) {\n"
            "    for (int c = 0; c < cols; c++) {\n"
            "        out[r][c] = in[r][c];\n"
            "    }\n"
            "}\n"
            "return 0;\n"
            "}\n"
        )
        outer_loop = (
            "for (int r = 0; r < rows; r++) {\n"
            "    for (int c = 0; c < cols; c++) {\n"
            "        out[r][c] = in[r][c];\n"
            "    }\n"
            "}"
        )
        inner_loop = (
            "for (int c = 0; c < cols; c++) {\n        out[r][c] = in[r][c];\n    }"
        )

        parallelizer = Parallelizer()
        parallelizer.cache_dir = tmp_path / "cache"
        monkeypatch.setattr(
            "accelera.src.parallelizer.parallelizer.extract_loops",
            lambda code: [SimpleNamespace()],
        )

        def fake_write_loops_to_json(loops, output_json):
            Path(output_json).write_text(
                json.dumps(
                    [
                        {
                            "code": outer_loop,
                            "start_line": 2,
                            "end_line": 6,
                            "type": "for",
                        },
                        {
                            "code": inner_loop,
                            "start_line": 3,
                            "end_line": 5,
                            "type": "for",
                        },
                    ]
                )
            )
            return True

        monkeypatch.setattr(
            "accelera.src.parallelizer.parallelizer.write_loops_to_json",
            fake_write_loops_to_json,
        )
        monkeypatch.setattr(
            "accelera.src.parallelizer.parallelizer.extract_features",
            lambda code: {"dummy": True},
        )
        monkeypatch.setattr(parallelizer, "_classify", lambda code: "parallel_for")

        parallelizer.parallelize(str(source_file), output_dir=tmp_path)

        output = (tmp_path / "parallelized_nested.c").read_text()
        assert output.count("#pragma omp parallel for") == 1
        assert "collapse(2)" in output


class TestExtractLoops:
    @pytest.fixture
    def simple_cpp_code(self):
        return """
        int main() {
            for (int i = 0; i < 10; i++) {
                // Simple loop
            }
            return 0;
        }
        """

    @pytest.fixture
    def multiple_loops_cpp_code(self):
        return """
        int main() {
            for (int i = 0; i < 10; i++) {}
            
            int j = 0;
            while (j < 5) {
                j++;
            }
            
            return 0;
        }
        """

    def test_extract_loops_simple_file(self, simple_cpp_code):
        loops = extract_loops(simple_cpp_code)

        assert isinstance(loops, list)
        assert len(loops) >= 1
        assert hasattr(loops[0], "type")
        assert hasattr(loops[0], "start_line")
        assert hasattr(loops[0], "end_line")
        assert hasattr(loops[0], "code")

    def test_extract_loops_multiple_loops(self, multiple_loops_cpp_code):
        loops = extract_loops(multiple_loops_cpp_code)

        assert isinstance(loops, list)
        assert len(loops) >= 2

    def test_extract_loops_empty_file(self):
        loops = extract_loops("")
        assert isinstance(loops, list)
        assert len(loops) == 0


class TestWriteLoopsToJson:
    @pytest.fixture
    def real_loops(self):
        content = """
        int main() {
            for (int i = 0; i < 10; i++) {}
            
            int j = 0;
            while (j < 5) {
                j++;
            }
            
            return 0;
        }
        """
        return extract_loops(content)

    def test_write_loops_to_json_success(self, real_loops):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_json = f.name

        try:
            result = write_loops_to_json(real_loops, temp_json)
            assert result is True
            assert os.path.exists(temp_json)

            with open(temp_json, "r") as f:
                content = f.read()
                assert "start_line" in content
                assert "end_line" in content
        finally:
            if os.path.exists(temp_json):
                os.remove(temp_json)

    def test_write_loops_to_json_empty_list(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_json = f.name

        try:
            result = write_loops_to_json([], temp_json)
            assert result is True
            assert os.path.exists(temp_json)
        finally:
            if os.path.exists(temp_json):
                os.remove(temp_json)
