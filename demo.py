import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from aistudio.src.core.pipeline import Pipeline


@pytest.fixture
def sample_data():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    test_data = np.array([[1.5, 1.5]])
    return X, y, test_data


def test_pipeline_predictions(sample_data):
    X, y, test_data = sample_data

    p = Pipeline()
    model = p.add("logreg", LogisticRegression(random_state=42))
    p.add("predict", model.predict, test_data)

    predictions = p(X, y)

    # Compare with manual implementation
    manual_model = LogisticRegression(random_state=42)
    manual_model.fit(X, y)
    expected = manual_model.predict(test_data)

    np.testing.assert_array_equal(predictions, expected)


def test_graph_consistency(sample_data):
    X, y, test_data = sample_data

    p = Pipeline()
    model = p.add("logreg", LogisticRegression(random_state=42))
    p.add("predict", model.predict, test_data)

    # Multiple calls should produce same results
    pred1 = p(X, y)
    pred2 = p(X, y)

    np.testing.assert_array_equal(pred1, pred2)
