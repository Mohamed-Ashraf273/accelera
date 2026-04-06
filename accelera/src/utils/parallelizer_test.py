import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from accelera.src.utils.parallelizer import Parallelizer


class TestParallelizer:
    def test_classify_returns_prediction(self, monkeypatch):
        parallelizer = Parallelizer()

        class DummyResponse:
            status_code = 200

            def json(self):
                return {"result": "parallel_for"}

        monkeypatch.setattr(
            "accelera.src.utils.parallelizer.requests.post",
            lambda url, json: DummyResponse(),
        )

        result = parallelizer._classify(np.array([1.0, 2.0], dtype=np.float32))

        assert result == "parallel_for"

    def test_generate_omp_pragma_with_loop_none_returns_loop(self):
        parallelizer = Parallelizer()
        loop_code = "for (int i = 0; i < n; ++i) {\n    sum += a[i];\n}"

        result = parallelizer._generate_omp_pragma_with_loop(loop_code, "none")

        assert result == loop_code

    def test_generate_omp_pragma_with_loop_adds_validated_pragma(
        self, monkeypatch
    ):
        parallelizer = Parallelizer()

        class DummyResponse:
            status_code = 200

            def json(self):
                return {"pragma": "omp parallel for"}

        monkeypatch.setattr(
            "accelera.src.utils.parallelizer.requests.post",
            lambda url, json, timeout: DummyResponse(),
        )
        monkeypatch.setattr(
            "accelera.src.utils.parallelizer.validate_pragma",
            lambda pragma: f"#pragma {pragma}",
        )

        result = parallelizer._generate_omp_pragma_with_loop(
            "for (int i = 0; i < n; ++i) {}", "parallel_for"
        )

        assert result.startswith("#pragma omp parallel for\n")
        assert "for (int i = 0; i < n; ++i) {}" in result

    def test_parallelize_writes_parallelized_output(
        self, monkeypatch, tmp_path
    ):
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
            "accelera.src.utils.parallelizer.extract_loops",
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
            "accelera.src.utils.parallelizer.write_loops_to_json",
            fake_write_loops_to_json,
        )
        monkeypatch.setattr(
            "accelera.src.utils.parallelizer.extract_features",
            lambda code: {"dummy": True},
        )
        monkeypatch.setattr(
            "accelera.src.utils.parallelizer.vectorize_features",
            lambda features: np.array([1.0], dtype=np.float32),
        )
        monkeypatch.setattr(parallelizer, "_classify", lambda embedding: "omp")
        monkeypatch.setattr(
            parallelizer,
            "_generate_omp_pragma_with_loop",
            lambda code, cls: f"#pragma omp parallel for\n{code}",
        )

        formatted_files = []
        monkeypatch.setattr(
            "accelera.src.utils.parallelizer.format_cpp_file",
            lambda path: formatted_files.append(Path(path)),
        )

        result = parallelizer.parallelize(str(source_file))

        output_file = tmp_path / "parallelized_sample.c"
        assert result is None
        assert output_file.exists()
        assert "#pragma omp parallel for" in output_file.read_text()
        assert formatted_files == [output_file]
