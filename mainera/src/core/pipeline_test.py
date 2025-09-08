import numpy as np
from sklearn.linear_model import LogisticRegression

from mainera.src.core.pipeline import Pipeline


def test_simple_preprocessing_correctness():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    y = np.array([0, 0, 1, 1])
    test_data = np.array([[4, 5]], dtype=np.float64)

    p = Pipeline()
    p.preprocess("scale", lambda x: x * 2.0)
    p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
    p.predict("pred", test_data)
    pipeline_result = p(X, y)

    X_scaled = X * 2.0
    test_scaled = test_data * 2.0
    manual_model = LogisticRegression(random_state=42, max_iter=1000)
    manual_model.fit(X_scaled, y)
    manual_correct_behavior = manual_model.predict(test_scaled)

    pipeline_pred = (
        pipeline_result[0]
        if isinstance(pipeline_result, list)
        else pipeline_result
    )
    assert np.array_equal(pipeline_pred, manual_correct_behavior)

    assert True


def test_multiple_preprocessing_steps():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    y = np.array([0, 0, 1, 1])
    test_data = np.array([[4, 5]], dtype=np.float64)

    p = Pipeline()
    p.preprocess("scale", lambda x: x * 2.0)
    p.preprocess("shift", lambda x: x + 1.0)
    p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
    p.predict("pred", test_data)
    pipeline_result = p(X, y)

    X_step1 = X * 2.0
    X_step2 = X_step1 + 1.0

    test_step1 = test_data * 2.0
    test_step2 = test_step1 + 1.0

    manual_model = LogisticRegression(random_state=42, max_iter=1000)
    manual_model.fit(X_step2, y)
    manual_result = manual_model.predict(test_step2)

    assert pipeline_result is not None
    pipeline_pred = (
        pipeline_result[0]
        if isinstance(pipeline_result, list)
        else pipeline_result
    )
    assert np.array_equal(pipeline_pred, manual_result)


def test_model_prediction_correctness():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    y = np.array([0, 0, 1, 1])
    test_data = np.array([[4, 5]], dtype=np.float64)

    p = Pipeline()
    p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
    p.predict("pred", test_data)
    pipeline_result = p(X, y)

    manual_model = LogisticRegression(random_state=42, max_iter=1000)
    manual_model.fit(X, y)
    manual_result = manual_model.predict(test_data)

    assert pipeline_result is not None
    pipeline_pred = (
        pipeline_result[0]
        if isinstance(pipeline_result, list)
        else pipeline_result
    )
    assert np.array_equal(pipeline_pred, manual_result)


def test_preprocessing_model_chain_correctness():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    y = np.array([0, 0, 1, 1])
    test_data = np.array([[4, 5]], dtype=np.float64)

    p = Pipeline()
    p.preprocess("scale", lambda x: x / 10.0)
    p.model("lr", LogisticRegression(random_state=42, max_iter=1000))
    p.predict("pred", test_data)
    pipeline_result = p(X, y)

    X_scaled = X / 10.0
    test_scaled = test_data / 10.0
    manual_model = LogisticRegression(random_state=42, max_iter=1000)
    manual_model.fit(X_scaled, y)
    manual_result = manual_model.predict(test_scaled)

    assert pipeline_result is not None
    pipeline_pred = (
        pipeline_result[0]
        if isinstance(pipeline_result, list)
        else pipeline_result
    )
    assert np.array_equal(pipeline_pred, manual_result)
