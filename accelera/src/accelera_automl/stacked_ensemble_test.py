import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

from accelera.src.accelera_automl.stacked_ensemble import ClassifierAdapter


def test_classifier_adapter_works_with_bagging_predict_proba():
    X, y = make_classification(
        n_samples=40,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=0,
    )

    estimator = BaggingClassifier(
        estimator=ClassifierAdapter(LogisticRegression()),
        n_estimators=3,
        random_state=0,
    )

    estimator.fit(X, y)
    proba = estimator.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
