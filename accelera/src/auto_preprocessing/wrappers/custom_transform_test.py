import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from accelera.src.auto_preprocessing.wrappers.frequency_encoder_transform import (
    FrequencyEncoderTransform,
)


class TestCustomTransform:
    def test_frequency_encoder_transform(self):
        data = pd.DataFrame(
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
        transformer.fit(data)
        transformed_data_data = transformer.transform(data)
        assert "color" in transformer.mapping_
        assert "size" in transformer.mapping_
        assert transformer.mapping_["color"] == {
            "blue": 0.5,
            "red": 0.3333333333333333,
            "green": 0.16666666666666666,
        }
        assert transformer.mapping_["size"] == {"S": 0.5, "L": 0.5}
        assert np.array_equal(transformed_data_data, expected_output.values)

    def test_frequency_encoder_transform_unseen_category(self):
        training_data = pd.DataFrame(
            {
                "color": ["red", "blue", "red", "green", "blue", "blue"],
                "size": ["S", "L", "L", "S", "S", "L"],
            }
        )
        testing_data = pd.DataFrame(
            {"color": ["red", "yellow"], "size": ["S", "XL"]}
        )
        expected_output = pd.DataFrame(
            {"color": [0.3333333333333333, 0], "size": [0.5, 0]}
        )
        transformer = FrequencyEncoderTransform()
        transformer.fit(training_data)
        transformed_data = transformer.transform(testing_data)
        assert np.array_equal(transformed_data, expected_output.values)

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
        transformer = ColumnTransformer(
            [("freq_encoder", pipeline, ["color", "size"])]
        )
        transformed_data = transformer.fit_transform(df)
        assert np.array_equal(transformed_data, expected_output.values)
