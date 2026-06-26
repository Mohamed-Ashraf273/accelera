from types import SimpleNamespace

import numpy as np
from joblib import Parallel
from joblib import delayed
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

from accelera.src.accelera_automl.utils import log_ensemble_structure
from accelera.src.accelera_automl.utils import log_forward_selection_step
from accelera.src.accelera_automl.utils import score_predictions


def softmax(logits):
    stable_numbers = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(stable_numbers)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


# use this class to make the model output prob
class ClassifierAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.estimator = clone(self.model)
        self.estimator.fit(X, y)
        self.classes_ = getattr(
            self.estimator,
            "classes_",
            getattr(self.estimator, "classes", np.unique(y)),
        )
        self.classes = self.classes_

        return self

    def predict_proba(self, X):
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X)
            return np.asarray(proba, dtype=float)

        if hasattr(self.estimator, "decision_function"):
            decision = np.asarray(self.estimator.decision_function(X), dtype=float)
            if decision.ndim == 1:
                positive = 1.0 / (1.0 + np.exp(-decision))  # sigmoid
                return np.column_stack([1.0 - positive, positive])

            return softmax(decision)

        predictions = np.asarray(self.estimator.predict(X))
        proba = np.zeros((predictions.shape[0], len(self.classes_)), dtype=float)
        for idx, class_label in enumerate(self.classes_):
            proba[:, idx] = (predictions == class_label).astype(float)
        return proba

    def predict(self, X):
        probability = self.predict_proba(X)
        return self.classes_[np.argmax(probability, axis=1)]


class StackedEnsembleClassifier(BaseEstimator, ClassifierMixin):
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
        scoring="accuracy",
        min_base_models=3,
        selection_tolerance=1e-4,
        verbose=0,
    ):
        self.base_estimators = base_estimators
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

    def make_bagged_model(self, model, n_jobs):
        adapted_estimator = ClassifierAdapter(model)
        bagging_kwargs = {
            "n_estimators": self.bagging_n_estimators,
            "random_state": self.random_state,
            "n_jobs": n_jobs,
        }

        return BaggingClassifier(estimator=adapted_estimator, **bagging_kwargs)

    def fit_base_model(self, model_name, model, X, y, splitter):
        inner_n_jobs = self.inner_n_jobs
        bagged_model = self.make_bagged_model(model, n_jobs=inner_n_jobs)
        oof_proba = cross_val_predict(
            bagged_model,
            X,
            y,
            cv=splitter,
            method="predict_proba",
            n_jobs=inner_n_jobs,
        )
        fitted_bagged_model = self.make_bagged_model(model, n_jobs=inner_n_jobs)
        fitted_bagged_model.fit(X, y)
        return model_name, fitted_bagged_model, np.asarray(oof_proba, dtype=float)

    def stack_features(self, proba):
        proba = np.asarray(proba, dtype=float)

        if proba.shape[1] == 2:
            return proba[:, [1]]
        return proba

    def combine_meta_features(
        self,
        X,
        prediction,
    ):
        prediction_matrix = np.hstack(prediction)
        if not self.include_original_features_in_meta:
            return prediction_matrix

        if sparse.issparse(X):
            return sparse.hstack(
                [X, sparse.csr_matrix(prediction_matrix)], format="csr"
            )

        original_matrix = np.asarray(X)
        return np.hstack([original_matrix, prediction_matrix])

    def evaluate_meta_subset(self, X, y, selected):
        X_train = [
            self.stack_features(predictions) for _, _, predictions in selected
        ]
        stack_X_train = self.combine_meta_features(X, X_train)
        splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

        meta_estimator = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            penalty="l2",
            max_iter=1000,
            random_state=self.random_state,
        )
        meta_oof_proba = cross_val_predict(
            ClassifierAdapter(meta_estimator),
            stack_X_train,
            y,
            cv=splitter,
            method="predict_proba",
            n_jobs=self.inner_n_jobs,
        )

        return score_predictions(
            self.classes_,
            self.scoring,
            y,
            np.asarray(meta_oof_proba, dtype=float),
        )

    def forward_select_base_models(self, X, y, base_results):
        remaining = list(base_results)
        selected = []
        best_single = max(
            remaining,
            key=lambda item: score_predictions(
                self.classes_,
                self.scoring,
                y,
                np.asarray(item[2], dtype=float),
            ),
        )
        selected.append(best_single)
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
        self.classes_ = np.unique(y)
        self.classes = self.classes_
        splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )
        base_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_base_model)(base_name, base_estimator, X, y, splitter)
            for base_name, base_estimator in self.base_estimators
        )

        self.selected_names, self.score_result = self.forward_select_base_models(
            X, y, base_results
        )
        selected_name_set = set(self.selected_names)
        selected_results = [
            result for result in base_results if result[0] in selected_name_set
        ]

        self.base_models = [
            (base_name, fitted_model)
            for base_name, fitted_model, _ in selected_results
        ]

        self.base_model_names = [base_name for base_name, _, _ in selected_results]

        base_features = [
            self.stack_features(predictions)
            for _, _, predictions in selected_results
        ]
        stack_train_X = self.combine_meta_features(X, base_features)

        meta_name = "LogisticRegression"
        meta_estimator = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            penalty="l2",
            max_iter=1000,
            random_state=self.random_state,
        )

        self.meta_modelname = meta_name
        self.meta_model = ClassifierAdapter(meta_estimator)
        self.meta_model.fit(stack_train_X, y)
        if self.verbose > 0:
            log_ensemble_structure(self.base_model_names, self.meta_modelname)
        self.forward_selection_ = SimpleNamespace(score=float(self.score_result))

        return self

    def predict_proba(self, X):
        stack_features = self.build_stack_features(X)
        return np.asarray(self.meta_model.predict_proba(stack_features), dtype=float)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def build_stack_features(self, X):
        blocks = [
            self.stack_features(model.predict_proba(X))
            for _, model in self.base_models
        ]
        return self.combine_meta_features(X, blocks)
