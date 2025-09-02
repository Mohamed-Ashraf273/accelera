import numpy as np
from sklearn.linear_model import LogisticRegression

from aistudio.src.core.pipeline import Pipeline


def sample_data():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    test_data = np.array([[1.5, 1.5]])
    return X, y, test_data


X, y, test_data = sample_data()

p = Pipeline()

# p.branch([
#     ("logreg", LogisticRegression(random_state=42)),
#     ("predict", model.predict, test_data)
# ])
model = p.add("logreg", LogisticRegression(random_state=42))
p.add("predict", model.predict)

predictions = p(X, y)
print(predictions)
# # Compare with manual implementation
# manual_model = LogisticRegression(random_state=42)
# manual_model.fit(X, y)
# expected = manual_model.predict(test_data)

# np.testing.assert_array_equal(predictions, expected)
