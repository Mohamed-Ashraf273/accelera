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

p.model("logreg", LogisticRegression(random_state=42))
p.model("svc", SVC(probability=True, random_state=42))
p.branch("branch", "logreg", "svc")
p.predict("predict", test_data)

print("Predictions:", p(X, y))
