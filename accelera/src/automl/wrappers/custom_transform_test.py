import pandas as pd
import numpy as np
from accelera.src.automl.wrappers.flatten_1d_transform import Flatten1DTransform
from accelera.src.automl.wrappers.frequency_encoder_transform import (
    FrequencyEncoderTransform,
)
from accelera.src.automl.wrappers.IQR_transform import IQRTransform
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class TestCustomTransform:
    def test_frequency_encoder_transform(self):
        df = pd.DataFrame(
            {
                "color": ["red", "blue", "red", "green", "blue", "blue"],
                "size": ["S", "L", "L", "S", "S", "L"],
            }
        )
        expected_output = pd.DataFrame(
            {
                "color": [
                    0.3333333333333333,
                    0.5,
                    0.3333333333333333,
                    0.16666666666666666,
                    0.5,
                    0.5,
                ],
                "size": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            }
        )
        transformer = FrequencyEncoderTransform()
        transformer.fit(df)
        transformed = transformer.transform(df)
        assert "color" in transformer.mapping_
        assert "size" in transformer.mapping_
        assert transformer.mapping_["color"] == {
            "blue": 0.5,
            "red": 0.3333333333333333,
            "green": 0.16666666666666666,
        }
        assert transformer.mapping_["size"] == {"S": 0.5, "L": 0.5}
        np.testing.assert_almost_equal(transformed, expected_output.values)

    def test_frequency_encoder_transform_unseen(self):
        df_train = pd.DataFrame(
            {
                "color": ["red", "blue", "red", "green", "blue", "blue"],
                "size": ["S", "L", "L", "S", "S", "L"],
            }
        )
        df_test = pd.DataFrame({"color": ["red", "yellow"], "size": ["S", "XL"]})
        expected_output = pd.DataFrame(
            {"color": [0.3333333333333333, 0], "size": [0.5, 0]}
        )
        transformer = FrequencyEncoderTransform()
        transformer.fit(df_train)
        transformed = transformer.transform(df_test)
        np.testing.assert_almost_equal(transformed, expected_output.values)

    def test_frequency_encoder_transform_pipeline(self):
        df = pd.DataFrame(
            {
                "color": ["red", "blue", "red", "green", "blue", "blue"],
                "size": ["S", "L", "L", "S", "S", "L"],
            }
        )
        expected_output = pd.DataFrame(
            {
                "color": [
                    0.3333333333333333,
                    0.5,
                    0.3333333333333333,
                    0.16666666666666666,
                    0.5,
                    0.5,
                ],
                "size": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            }
        )
        pipeline = Pipeline([("freq_encoder", FrequencyEncoderTransform())])
        column_transformer = ColumnTransformer(
            [("freq_encoder", pipeline, ["color", "size"])]
        )
        transformed = column_transformer.fit_transform(df)
        np.testing.assert_almost_equal(transformed, expected_output.values)

    def test_flatten_1d_transform(self):
        data = np.array([["Hello"], ["World"]])
        expected_output = np.array(["Hello", "World"])
        transformer = Flatten1DTransform(func=lambda x: x.ravel())
        transformed = transformer.fit_transform(data)
        assert transformed.shape == expected_output.shape
        np.testing.assert_array_equal(transformed, expected_output)

    def test_flatten_1d_transform_pipeline(self):
        data = pd.DataFrame({"col": ["Hello", "World"]})

        expected_output = np.array([["Hello"], ["World"]])
        pipeline = Pipeline(
            [
                (
                    "flatten",
                    Flatten1DTransform(func=lambda x: x.values.ravel()[:, np.newaxis]),
                )
            ]
        )
        column_transformer = ColumnTransformer([("flatten", pipeline, ["col"])])
        transformed = column_transformer.fit_transform(data)
        assert transformed.shape == expected_output.shape
        np.testing.assert_array_equal(transformed, expected_output)

    def test_iqr_transform(self):
        data = pd.DataFrame(
            {"col": [1, 2, 3, 4, 5, 100], "col2": [10, 20, 30, 40, 50, 1000]}
        )
        info = {
            "col": {"col_type": "numerical", "outliers_info": (2, 5)},
            "col2": {"col_type": "numerical", "outliers_info": (20, 50)},
        }
        expected_output = pd.DataFrame(
            {"col": [2, 2, 3, 4, 5, 5], "col2": [20, 20, 30, 40, 50, 50]}
        )
        transformer = IQRTransform(info=info, cols=["col", "col2"])
        output = transformer.fit_transform(data.values)
        np.testing.assert_array_equal(output, expected_output.values)

    def test_iqr_transform_pipeline_data_frame(self):
        data = pd.DataFrame(
            {"col": [1, 2, 3, 4, 5, 100], "col2": [10, 20, 30, 40, 50, 1000]}
        )
        info = {
            "col": {"col_type": "numerical", "outliers_info": (2, 5)},
            "col2": {"col_type": "numerical", "outliers_info": (20, 50)},
        }
        expected_output = pd.DataFrame(
            {"col": [2, 2, 3, 4, 5, 5], "col2": [20, 20, 30, 40, 50, 50]}
        )
        pipeline = Pipeline(
            [("iqr_transform", IQRTransform(info=info, cols=["col", "col2"]))]
        )
        column_transformer = ColumnTransformer(
            [("iqr_transform", pipeline, ["col", "col2"])]
        )
        output = column_transformer.fit_transform(data)
        np.testing.assert_array_equal(output, expected_output.values)

    def test_iqr_transform_pipeline_numpy(self):
        data = pd.DataFrame(
            {"col": [1, 2, 3, 4, 5, 100], "col2": [10, 20, 30, 40, 50, 1000]}
        )
        info = {
            "col": {"col_type": "numerical", "outliers_info": (2, 5)},
            "col2": {"col_type": "numerical", "outliers_info": (20, 50)},
        }
        expected_output = pd.DataFrame(
            {"col": [2, 2, 3, 4, 5, 5], "col2": [20, 20, 30, 40, 50, 50]}
        )
        pipeline = Pipeline(
            [("iqr_transform", IQRTransform(info=info, cols=["col", "col2"]))]
        )
        column_transformer = ColumnTransformer([("iqr_transform", pipeline, [0, 1])])
        output = column_transformer.fit_transform(data.values)
        np.testing.assert_array_equal(output, expected_output.values)
