import numpy as np
import pytest

from accelera.src.utils.array_utils import convert_to_array
from accelera.src.utils.array_utils import get_array_info
from accelera.src.utils.array_utils import validate_array_shape


class TestConvertToArray:
    def test_convert_none_returns_none(self):
        result = convert_to_array(None)
        assert result is None

    def test_convert_list_to_array(self):
        input_list = [1, 2, 3, 4]
        result = convert_to_array(input_list)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.asarray(input_list))

    def test_convert_with_dtype(self):
        input_list = [1.5, 2.7, 3.9]
        result = convert_to_array(input_list, dtype=np.int32)

        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, np.asarray(input_list, dtype=np.int32))


class TestValidateArrayShape:
    def test_validate_matching_shapes(self):
        array = np.array([[1, 2], [3, 4], [5, 6]])
        validate_array_shape(array, 3)

    def test_validate_mismatched_shapes_raises_error(self):
        array = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="must have the same number of samples"):
            validate_array_shape(array, 5)

    def test_validate_with_custom_names(self):
        array = np.array([1, 2, 3])
        names = ["predictions", "targets"]

        with pytest.raises(ValueError) as exc_info:
            validate_array_shape(array, 5, names)

        error_msg = str(exc_info.value)
        assert "predictions" in error_msg
        assert "targets" in error_msg


class TestGetArrayInfo:
    def test_get_info_none_input(self):
        """Test getting info for None input."""
        result = get_array_info(None)
        expected = {"shape": None, "dtype": None, "ndim": None, "size": None}
        assert result == expected

    def test_get_info_array(self):
        array = np.array([[1, 2, 3], [4, 5, 6]])
        result = get_array_info(array)

        assert result["shape"] == (2, 3)
        assert result["ndim"] == 2
        assert result["size"] == 6

    def test_get_info_list_input(self):
        result = get_array_info([1, 2, 3])

        assert result["shape"] == (3,)
        assert result["ndim"] == 1
        assert result["size"] == 3
