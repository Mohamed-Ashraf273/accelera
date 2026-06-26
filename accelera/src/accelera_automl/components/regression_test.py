from accelera.src.accelera_automl.components.regression import build_ard_regression
from accelera.src.accelera_automl.components.regression import (
    build_gradient_boosting,
)


def test_build_ard_regression_accepts_regression_evaluator_signature():
    estimator = build_ard_regression(
        {
            "alpha_1": 1e-6,
            "alpha_2": 1e-6,
            "fit_intercept": True,
            "lambda_1": 1e-6,
            "lambda_2": 1e-6,
            "max_iter": 100,
            "threshold_lambda": 10000.0,
            "tol": 1e-4,
        },
        random_state=42,
        n_jobs=1,
    )

    iteration_value = (
        estimator.max_iter if hasattr(estimator, "max_iter") else estimator.n_iter
    )
    assert iteration_value == 100


def test_build_gradient_boosting_omits_early_stopping_only_params_when_disabled():
    estimator = build_gradient_boosting(
        {
            "early_stop": "off",
            "l2_regularization": 1e-4,
            "learning_rate": 0.1,
            "loss": "squared_error",
            "max_bins": 255,
            "max_depth": 3,
            "max_iter": 50,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "scoring": "loss",
            "tol": 1e-7,
        },
        random_state=42,
        n_jobs=1,
    )

    assert estimator.early_stopping is False
    assert estimator.n_iter_no_change == 10
