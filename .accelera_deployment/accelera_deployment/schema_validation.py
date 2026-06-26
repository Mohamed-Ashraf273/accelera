from uuid import uuid4

import great_expectations as gx
import pandas as pd


class SchemaValidationError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(errors))


class InputSchema:
    def __init__(self, config=None):
        config = config or {}
        self.features = config.get("features", [])
        self.enabled = bool(self.features)
        self._batch_definition = None

        if self.enabled:
            suffix = uuid4().hex
            context = gx.get_context()
            data_source = context.data_sources.add_pandas(
                name=f"prediction_input_{suffix}"
            )
            data_asset = data_source.add_dataframe_asset(
                name=f"prediction_payload_{suffix}"
            )
            self._batch_definition = data_asset.add_batch_definition_whole_dataframe(
                f"prediction_batch_{suffix}"
            )

    def describe(self):
        return {
            "enabled": self.enabled,
            "features": self.features,
        }

    def validate(self, input_data):
        if not self.enabled:
            return input_data

        df = self.to_dataframe(input_data)
        errors = []
        expected_columns = [feature["name"] for feature in self.features]
        missing_columns = [col for col in expected_columns if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in expected_columns]

        if missing_columns:
            errors.append(f"missing columns {missing_columns}")
        if extra_columns:
            errors.append(f"unexpected columns {extra_columns}")
        if errors:
            raise SchemaValidationError(errors)

        df = df[expected_columns].copy()
        self.coerce_columns(df, errors)
        if errors:
            raise SchemaValidationError(errors)

        batch = self._batch_definition.get_batch(batch_parameters={"dataframe": df})

        for feature in self.features:
            column = feature["name"]
            expectations = []

            if feature.get("required", True):
                expectations.append(
                    gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
                )

            if "min" in feature or "max" in feature:
                ranges = {"column": column}
                if "min" in feature:
                    ranges["min_value"] = feature["min"]
                if "max" in feature:
                    ranges["max_value"] = feature["max"]
                expectations.append(
                    gx.expectations.ExpectColumnValuesToBeBetween(
                        **ranges,
                    )
                )

            if "allowed_values" in feature:
                expectations.append(
                    gx.expectations.ExpectColumnValuesToBeInSet(
                        column=column,
                        value_set=feature["allowed_values"],
                    )
                )

            for expectation in expectations:
                result = batch.validate(expectation)
                if not result.success:
                    expect = expectation.__class__.__name__
                    errors.append(f"error {column} {expect}")

        if errors:
            raise SchemaValidationError(errors)

        return df.values.tolist()

    def to_dataframe(self, input_data):
        columns = [feature["name"] for feature in self.features]

        if isinstance(input_data, pd.DataFrame):
            return input_data

        if isinstance(input_data, dict):
            return pd.DataFrame([input_data])

        if isinstance(input_data, list) and input_data:
            if all(not isinstance(item, (list, tuple, dict)) for item in input_data):
                return pd.DataFrame([input_data], columns=columns)

        return pd.DataFrame(input_data, columns=columns)

    def coerce_columns(self, df, errors):
        for feture in self.features:
            column = feture["name"]
            feature_type = feture.get("type", "number")
            non_null = df[column].notna()

            if feature_type in {"number", "integer"}:
                original = df[column].copy()
                df[column] = pd.to_numeric(df[column], errors="coerce")
                invalid = non_null & df[column].isna()
                if invalid.any():
                    errors.append(f"{column} not same as {feature_type}")
                    df[column] = original
                    continue

                if feature_type == "integer":
                    integer_values = df.loc[non_null, column] % 1 == 0
                    if not integer_values.all():
                        errors.append(f"{column} must be only intgers")
                    df[column] = df[column].astype("Int64")

            elif feature_type == "string":
                df.loc[non_null, column] = df.loc[non_null, column].astype(str)

            elif feature_type == "boolean":
                allowed = {True, False, "true", "false", "1", "0", 0, 1}
                invalid = non_null & ~df[column].isin(allowed)
                if invalid.any():
                    errors.append(f"{column} must be boolean")
                df[column] = df[column].replace(
                    {"true": True, "false": False, "1": True, "0": False}
                )

            else:
                errors.append(f"{column} unsupported type {feature_type!r}")
