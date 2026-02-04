import os
import tempfile
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from accelera.src.utils.code_utils import clean_response
from accelera.src.utils.code_utils import extract_loops
from accelera.src.utils.code_utils import format_cpp_file
from accelera.src.utils.code_utils import write_file
from accelera.src.utils.code_utils import write_loops_to_json


class TestExtractLoops:
    @pytest.fixture
    def simple_cpp_file(self):
        content = """
        int main() {
            for (int i = 0; i < 10; i++) {
                // Simple loop
            }
            return 0;
        }
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", delete=False
        ) as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.remove(temp_path)

    @pytest.fixture
    def multiple_loops_cpp_file(self):
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", delete=False
        ) as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_extract_loops_simple_file(self, simple_cpp_file):
        loops = extract_loops(simple_cpp_file)

        assert isinstance(loops, list)
        assert len(loops) >= 1
        assert hasattr(loops[0], "type")
        assert hasattr(loops[0], "start_line")
        assert hasattr(loops[0], "end_line")
        assert hasattr(loops[0], "code")

    def test_extract_loops_multiple_loops(self, multiple_loops_cpp_file):
        loops = extract_loops(multiple_loops_cpp_file)

        assert isinstance(loops, list)
        assert len(loops) >= 2

    def test_extract_loops_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", delete=False
        ) as f:
            f.write("")
            temp_path = f.name

        try:
            loops = extract_loops(temp_path)
            assert isinstance(loops, list)
            assert len(loops) == 0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", delete=False
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            loops = extract_loops(temp_path)
            yield loops
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

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


class TestCleanResponse:
    def test_clean_response_with_cpp_fence(self):
        response = "```cpp\nint main() { return 0; }\n```"
        cleaned = clean_response(response)
        assert cleaned == "int main() { return 0; }"

    def test_clean_response_with_generic_fence(self):
        response = "```\nint main() { return 0; }\n```"
        cleaned = clean_response(response)
        assert cleaned == "int main() { return 0; }"

    def test_clean_response_with_unwanted_prefixes(self):
        test_cases = [
            "Here's the C++ code:\nint main() {}",
            "C++ code:\nint main() {}",
        ]

        for response in test_cases:
            cleaned = clean_response(response)
            assert cleaned == "int main() {}"

    def test_clean_response_no_changes_needed(self):
        response = "int main() { return 0; }"
        cleaned = clean_response(response)
        assert cleaned == response


class TestFormatCppFile:
    @pytest.fixture
    def temp_cpp_file(self):
        content = "int main(){return 0;}"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", delete=False
        ) as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.remove(temp_path)

    @patch("subprocess.run")
    def test_format_cpp_file_success(self, mock_run, temp_cpp_file):
        mock_run.return_value = MagicMock(returncode=0)

        result = format_cpp_file(temp_cpp_file)
        assert result is True
        assert mock_run.call_count == 2

    @patch("subprocess.run")
    def test_format_cpp_file_clang_format_not_found(
        self, mock_run, temp_cpp_file
    ):
        mock_run.side_effect = FileNotFoundError()

        result = format_cpp_file(temp_cpp_file)
        assert result is False


class TestWriteFile:
    @pytest.fixture
    def temp_directory(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir

        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @patch("accelera.src.utils.code_utils.format_cpp_file")
    def test_write_file_success(self, mock_format, temp_directory):
        filename = os.path.join(temp_directory, "test.cpp")
        python_code = "dummy_code"
        response = "int main() { return 0; }"
        mock_format.return_value = True

        result = write_file(filename, python_code, response)

        assert result == "int main() { return 0; }"
        assert os.path.exists(filename)
        mock_format.assert_called_once_with(filename)

    def test_write_file_invalid_extension(self, temp_directory):
        filename = os.path.join(temp_directory, "test.txt")
        python_code = "dummy_code"
        response = "int main() { return 0; }"

        with pytest.raises(AssertionError, match="must have a .cpp extension"):
            write_file(filename, python_code, response)
