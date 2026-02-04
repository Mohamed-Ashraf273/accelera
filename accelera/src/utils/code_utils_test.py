import os
import tempfile
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from accelera.src.utils.code_utils import extract_loops
from accelera.src.utils.code_utils import format_cpp_file
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
