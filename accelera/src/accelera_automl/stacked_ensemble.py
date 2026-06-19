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
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict


def softmax(logits):
    stable_numbers = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(stable_numbers)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


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
        meta_estimators=None,
        cv=5,
        random_state=None,
        n_jobs=None,
        inner_n_jobs=1,
        bagging_n_estimators=5,
        include_original_features_in_meta=True,
        scoring="accuracy",
        verbose=0,
        min_base_models=3,
        selection_tolerance=1e-4,
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
        self.verbose = verbose
        self.min_base_models = max(1, int(min_base_models))
        self.selection_tolerance = float(selection_tolerance)

    def fit(self, X, y):
        self.classes = np.unique(y)
        splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

        base_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_base_model)(base_name, base_estimator, X, y, splitter)
            for base_name, base_estimator in self.base_estimators
        )
        if not base_results:
            raise RuntimeError("No base result exist.")

        self.selected_names, self.score_result = self.forward_select_base_models(
            X, np.asarray(y), base_results
        )

        selected_names = set(self.selected_names)
        selected_results = [
            (base_name, fitted_model, oof_proba)
            for base_name, fitted_model, oof_proba in base_results
            if base_name in selected_names
        ]
        if len(selected_results) < self.min_base_models:
            raise RuntimeError(
                "Forward selection did not produce the minimum number "
                "of base models."
            )

        self.base_models = [
            (base_name, fitted_model)
            for base_name, fitted_model, _ in selected_results
        ]
        self.base_model_names_ = [base_name for base_name, _, _ in selected_results]
        base_feature_blocks = [
            self.stack_features_from_proba(oof_proba)
            for base_name, fitted_model, oof_proba in selected_results
        ]
        stack_train_X = self.combine_meta_features(X, base_feature_blocks)

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

        self.forward_selection_ = SimpleNamespace(score=float(self.score_result))
        self.log_ensemble_structure()
        return self

    def predict_proba(self, X):
        stack_features = self.build_stack_features(X)
        return np.asarray(self.meta_model.predict_proba(stack_features), dtype=float)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]

    def fit_base_model(
        self,
        base_name,
        base_estimator,
        X,
        y,
        splitter,
    ):
        inner_n_jobs = self.inner_n_jobs
        bagged_model = self.make_bagged_model(base_estimator, n_jobs=inner_n_jobs)
        oof_proba = cross_val_predict(
            bagged_model,
            X,
            y,
            cv=splitter,
            method="predict_proba",
            n_jobs=inner_n_jobs,
        )
        fitted_bagged_model = self.make_bagged_model(
            base_estimator, n_jobs=inner_n_jobs
        )
        fitted_bagged_model.fit(X, y)
        return base_name, fitted_bagged_model, np.asarray(oof_proba, dtype=float)

    def forward_select_base_models(
        self,
        X,
        y,
        base_results,
    ):
        remaining = list(base_results)
        if not remaining:
            raise RuntimeError(
                "No base model results available for forward selection."
            )

        selected = []
        best_single = max(
            remaining, key=lambda item: self.score_predictions(y, item[2])
        )
        selected.append(best_single)
        remaining.remove(best_single)
        current_score = self.evaluate_meta_subset(X, y, selected)
        self.log_forward_selection_step(
            step=1,
            selected_names=[best_single[0]],
            score=current_score,
            improvement=None,
        )

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
            self.log_forward_selection_step(
                step=len(selected),
                selected_names=[name for name, _, _ in selected],
                score=current_score,
                improvement=improvement,
            )
        return [name for name, _, _ in selected], current_score

    def evaluate_meta_subset(
        self,
        X,
        y,
        selected_results,
    ):
        blocks = [
            self.stack_features_from_proba(oof_proba)
            for base_name, fitted_model, oof_proba in selected_results
        ]
        stack_train_X = self.combine_meta_features(X, blocks)
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
            stack_train_X,
            y,
            cv=splitter,
            method="predict_proba",
            n_jobs=self.inner_n_jobs,
        )
        return self.score_predictions(y, np.asarray(meta_oof_proba, dtype=float))

    def make_bagged_model(self, base_estimator, n_jobs=None):
        adapted_estimator = ClassifierAdapter(base_estimator)
        bagging_kwargs = {
            "n_estimators": self.bagging_n_estimators,
            "random_state": self.random_state,
            "n_jobs": n_jobs if n_jobs is not None else self.n_jobs,
        }
        try:
            return BaggingClassifier(estimator=adapted_estimator, **bagging_kwargs)
        except TypeError:
            return BaggingClassifier(
                base_estimator=adapted_estimator, **bagging_kwargs
            )

    def build_stack_features(self, X):
        blocks = []
        for base_name, model in self.base_models:
            proba = model.predict_proba(X)
            blocks.append(self.stack_features_from_proba(proba))
        return self.combine_meta_features(X, blocks)

    def log_ensemble_structure(self):
        if self.verbose <= 0:
            return
        print("[AutoML] Stacked ensemble summary")
        print(f"[AutoML] Selected base models: {', '.join(self.base_model_names_)}")
        print(f"[AutoML] Meta model: {self.meta_modelname}")
        print(
            f"[AutoML] Forward selection score: {self.forward_selection_.score:.6f} "
            f"(uses original features={self.include_original_features_in_meta})"
        )

    def log_forward_selection_step(
        self,
        *,
        step,
        selected_names,
        score,
        improvement,
    ):
        if self.verbose <= 0:
            return
        if improvement is None:
            print(
                f"[AutoML] Forward selection step {step}: "
                f"selected {selected_names[-1]} score={score:.6f}"
            )
            return
        print(
            f"[AutoML] Forward selection step {step}: "
            f"added {selected_names[-1]} score={score:.6f} "
            f"improvement={improvement:.6f}"
        )

    def score_predictions(self, y_true, proba):
        y_pred = self.classes[np.argmax(proba, axis=1)]
        scoring = self.scoring

        if scoring == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if scoring == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        if scoring == "f1":
            average = "binary" if len(self.classes) == 2 else "macro"
            return float(f1_score(y_true, y_pred, average=average))
        if scoring == "f1_macro":
            return float(f1_score(y_true, y_pred, average="macro"))
        if scoring == "f1_micro":
            return float(f1_score(y_true, y_pred, average="micro"))
        if scoring == "f1_weighted":
            return float(f1_score(y_true, y_pred, average="weighted"))
        if scoring == "precision":
            average = "binary" if len(self.classes) == 2 else "macro"
            return float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            )
        if scoring == "precision_macro":
            return float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            )
        if scoring == "precision_micro":
            return float(
                precision_score(y_true, y_pred, average="micro", zero_division=0)
            )
        if scoring == "precision_weighted":
            return float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            )
        if scoring == "recall":
            average = "binary" if len(self.classes) == 2 else "macro"
            return float(
                recall_score(y_true, y_pred, average=average, zero_division=0)
            )
        if scoring == "recall_macro":
            return float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            )
        if scoring == "recall_micro":
            return float(
                recall_score(y_true, y_pred, average="micro", zero_division=0)
            )
        if scoring == "recall_weighted":
            return float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            )
        if scoring == "roc_auc" and proba.shape[1] == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        if scoring == "average_precision" and proba.shape[1] == 2:
            return float(average_precision_score(y_true, proba[:, 1]))
        if scoring in {"neg_log_loss", "log_loss"}:
            return float(-log_loss(y_true, proba, labels=self.classes))
        return float(accuracy_score(y_true, y_pred))

    @staticmethod
    def stack_features_from_proba(proba):
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2:
            raise ValueError("Expected probability predictions to be a 2D array.")
        if proba.shape[1] == 2:
            return proba[:, [1]]
        return proba

    def combine_meta_features(
        self,
        X,
        prediction_blocks,
    ):
        prediction_matrix = np.hstack(prediction_blocks)
        if not self.include_original_features_in_meta:
            return prediction_matrix

        if sparse.issparse(X):
            return sparse.hstack(
                [X, sparse.csr_matrix(prediction_matrix)], format="csr"
            )

        original_matrix = np.asarray(X)
        return np.hstack([original_matrix, prediction_matrix])
