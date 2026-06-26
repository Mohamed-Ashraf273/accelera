from types import SimpleNamespace

import numpy as np
from joblib import Parallel
from joblib import delayed
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

from accelera.src.accelera_automl.utils import log_ensemble_structure
from accelera.src.accelera_automl.utils import log_forward_selection_step


class StackedEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        *,
        base_estimators,
        cv=5,
        random_state=None,
        n_jobs=None,
        inner_n_jobs=1,
        bagging_n_estimators=5,
        include_original_features_in_meta=True,
        scoring="r2",
        min_base_models=3,
        selection_tolerance=1e-4,
        meta_estimators=None,
        verbose=0,
    ):
        self.base_estimators = base_estimators
        self.meta_estimators = meta_estimators
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.inner_n_jobs = inner_n_jobs
        self.bagging_n_estimators = bagging_n_estimators
        self.include_original_features_in_meta = include_original_features_in_meta
        self.scoring = scoring
        self.min_base_models = min_base_models
        self.selection_tolerance = selection_tolerance
        self.verbose = verbose

    def combine_meta_features(self, X, selected):
        prediction_matrix = np.hstack(selected)
        if not self.include_original_features_in_meta:
            return prediction_matrix
        if sparse.issparse(X):
            return sparse.hstack(
                [X, sparse.csr_matrix(prediction_matrix)], format="csr"
            )
        return np.hstack([np.asarray(X), prediction_matrix])

    def evaluate_meta_subset(
        self,
        X,
        y,
        selected_results,
    ):
        X_train = [
            np.asarray(oof_pred, dtype=float).reshape(-1, 1)
            for _, _, oof_pred in selected_results
        ]
        stack_X_train = self.combine_meta_features(X, X_train)
        splitter = KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        meta_estimator = Ridge(random_state=self.random_state, alpha=1.0)
        meta_oof_pred = cross_val_predict(
            clone(meta_estimator),
            stack_X_train,
            y,
            cv=splitter,
            method="predict",
            n_jobs=self.inner_n_jobs,
        )
        return self.score_predictions(y, np.asarray(meta_oof_pred, dtype=float))

    def make_bagged_model(self, base_estimator, n_jobs=None):
        bagging_kwargs = {
            "n_estimators": self.bagging_n_estimators,
            "random_state": self.random_state,
            "n_jobs": n_jobs,
        }

        return BaggingRegressor(estimator=clone(base_estimator), **bagging_kwargs)

    def fit_base_model(self, base_name, base_estimator, X, y, splitter):
        bagged_model = self.make_bagged_model(
            base_estimator, n_jobs=self.inner_n_jobs
        )
        predictions = cross_val_predict(
            bagged_model,
            X,
            y,
            cv=splitter,
            method="predict",
            n_jobs=self.inner_n_jobs,
        )
        fitted_bagged_model = self.make_bagged_model(
            base_estimator, n_jobs=self.inner_n_jobs
        )
        fitted_bagged_model.fit(X, y)
        return base_name, fitted_bagged_model, np.asarray(predictions, dtype=float)

    def forward_select_base_models(
        self,
        X,
        y,
        base_results,
    ):
        remaining = list(base_results)
        best_single = max(
            remaining,
            key=lambda item: self.score_predictions(
                y,
                np.asarray(item[2], dtype=float),
            ),
        )
        selected = [best_single]
        remaining.remove(best_single)
        current_score = self.evaluate_meta_subset(X, y, selected)
        while remaining:
            best_candidate = None
            best_candidate_score = float("-inf")
            for candidate in remaining:
                trial_selected = selected + [candidate]
                trial_score = self.evaluate_meta_subset(X, y, trial_selected)
                if trial_score > best_candidate_score:
                    best_candidate_score = trial_score
                    best_candidate = candidate

            if best_candidate is None:
                break

            improvement = best_candidate_score - current_score
            if (
                len(selected) >= self.min_base_models
                and improvement <= self.selection_tolerance
            ):
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)
            current_score = best_candidate_score
            if self.verbose > 0:
                log_forward_selection_step(
                    [name for name, _, _ in selected],
                    current_score,
                )
        return [name for name, _, _ in selected], current_score

    def fit(self, X, y):
        splitter = KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        base_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_base_model)(base_name, base_estimator, X, y, splitter)
            for base_name, base_estimator in self.base_estimators
        )

        self.selected_names, self.score_result = self.forward_select_base_models(
            X, np.asarray(y), base_results
        )

        self.base_models = [
            (base_name, fitted_model) for base_name, fitted_model, _ in base_results
        ]
        self.base_model_names = [base_name for base_name, _, _ in base_results]
        base_prediction_blocks = [
            np.asarray(oof_pred, dtype=float).reshape(-1, 1)
            for _, _, oof_pred in base_results
        ]
        stack_train_X = self.combine_meta_features(X, base_prediction_blocks)

        meta_name, meta_estimator = (
            "meta_ridge_regressor",
            Ridge(random_state=self.random_state, alpha=1.0),
        )
        self.meta_model_name = meta_name
        self.meta_model = clone(meta_estimator)
        self.meta_model.fit(stack_train_X, y)
        if self.verbose > 0:
            log_ensemble_structure(self.base_model_names, self.meta_model_name)
        self.forward_selection_ = SimpleNamespace(score=float(self.score_result))

        return self

    def predict(self, X):
        stack_features = self.build_stack_features(X)
        return np.asarray(self.meta_model.predict(stack_features), dtype=float)

    def build_stack_features(self, X):
        blocks = [
            np.asarray(model.predict(X), dtype=float).reshape(-1, 1)
            for _, model in self.base_models
        ]
        return self.combine_meta_features(X, blocks)

    def score_predictions(self, y_true, prediction):
        if self.scoring in {"r2", None}:
            return float(r2_score(y_true, prediction))
        if self.scoring in {
            "neg_root_mean_squared_error",
            "root_mean_squared_error",
        }:
            return float(-np.sqrt(mean_squared_error(y_true, prediction)))
        if self.scoring in {"neg_mean_squared_error", "mean_squared_error"}:
            return float(-mean_squared_error(y_true, prediction))
        if self.scoring in {
            "neg_mean_absolute_error",
            "mean_absolute_error",
            "median_absolute_error",
        }:
            return float(-mean_absolute_error(y_true, prediction))
        return float(r2_score(y_true, prediction))


def make_voting_regressor(estimators):
    return VotingRegressor(estimators=estimators)
