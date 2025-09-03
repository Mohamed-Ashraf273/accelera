import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from aistudio.src.core.pipeline import Pipeline


def sample_data():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    test_data = np.array([[1.5, 1.5]])
    return X, y, test_data


X, y, test_data = sample_data()

p = Pipeline()

p.branch(
    "branch1",
    p.preprocess("power", lambda x: x**2, branch=True),
    p.preprocess("power", lambda x: x**3, branch=True),
)
p.preprocess("power", lambda x: x**4)
p.branch(
    "branch1",
    p.model("logreg", LogisticRegression(random_state=42), branch=True),
    p.model("svc", SVC(probability=True, random_state=42), branch=True),
)
p.predict("predict", test_data)
p.serialize("pipeline_with merge.xml")
print("Predictions:", p(X, y))
