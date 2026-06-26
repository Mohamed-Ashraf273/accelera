from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from accelera.src.accelera_automl.stacked_ensemble_regression import (
    StackedEnsembleRegressor,
)


def test_verbose_fit_records_and_logs_forward_selection_score(capsys):
    X, y = make_regression(
        n_samples=40,
        n_features=5,
        n_informative=4,
        noise=0.1,
        random_state=42,
    )
    ensemble = StackedEnsembleRegressor(
        base_estimators=[
            ("linear", LinearRegression()),
            ("tree", DecisionTreeRegressor(random_state=42)),
            ("neighbors", KNeighborsRegressor()),
        ],
        meta_estimators=[],
        cv=2,
        random_state=42,
        n_jobs=1,
        inner_n_jobs=1,
        bagging_n_estimators=1,
        include_original_features_in_meta=False,
        verbose=1,
    )

    ensemble.fit(X, y)
    output = capsys.readouterr().out

    assert ensemble.forward_selection_.score == ensemble.score_result
    assert "Forward selection selected" in output
