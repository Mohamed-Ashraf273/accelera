import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from accelera.src.accelera_automl.base_evaluation import BaseEvaluation
from accelera.src.accelera_automl.components import get_classification_components
from accelera.src.accelera_automl.components import get_regression_components


class ClassificationEvaluator(BaseEvaluation):
    def __init__(
        self,
        *,
        cv=5,
        scoring="accuracy",
        random_state=None,
        n_jobs=None,
        balance_classes=False,
    ):
        super().__init__(
            cv=cv, scoring=scoring, random_state=random_state, n_jobs=n_jobs
        )
        self.balance_classes = balance_classes

    def build_estimator(self, model_name, params, trial_specs):
        cls_components = get_classification_components()
        if model_name not in cls_components:
            raise ValueError(f"unsupported classification model `{model_name}`.")

        cls_component = cls_components[model_name]
        est = cls_component.build_estimator(
            params, self.random_state, self.n_jobs, self.balance_classes
        )
        self.apply_model_budget(est, model_name, params, trial_specs)
        return est

    def get_candidate_preprocessors(
        self,
        model_name,
        params,
    ):
        candidates = ["none"]

        if model_name in {
            "svc",
            "knn",
            "lda",
            "liblinear_svc",
            "passive_aggressive",
            "qda",
            "sgd",
            "ridge_classifier",
            "mlp",
            "gaussian_process",
        }:
            candidates.extend(["standard", "robust"])
        elif model_name == "logistic_regression":
            solver = params.get("solver")
            if solver in {"lbfgs", "saga", "liblinear"}:
                candidates.extend(["standard", "robust", "minmax"])

        # preserve order while removing duplicates.
        return list(dict.fromkeys(candidates))

    def convert_score_to_cost(self, score):
        if self.scoring == "accuracy":
            return float(1.0 - score)
        return float(-score)

    def get_cv_splitter(self, cv_num_of_splits):
        return StratifiedKFold(
            n_splits=cv_num_of_splits, shuffle=True, random_state=self.random_state
        )

    def make_subsample_splitter(self, target_size):
        return StratifiedShuffleSplit(
            n_splits=1, train_size=target_size, random_state=self.random_state
        )

    def return_failure_score(self):
        return 0.0

    def resolve_cv_folds(self, y, requested_folds):
        values = np.asarray(y)
        _, counts = np.unique(values, return_counts=True)
        if counts.size == 0:
            return 2
        # each fold must contain at least one sample from every class.
        max_supported = int(np.min(counts))
        return max(2, min(requested_folds, max_supported))


class RegressionEvaluator(BaseEvaluation):
    def __init__(
        self,
        *,
        cv=5,
        scoring="r2",
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def build_estimator(self, model_name, params, trial_specs):
        components = get_regression_components()
        if model_name not in components:
            raise ValueError(f"unsupported regression model `{model_name}`.")
        component = components[model_name]
        estimator = component.build_estimator(params, self.random_state, self.n_jobs)
        self.apply_model_budget(estimator, model_name, params, trial_specs)
        return estimator

    def get_candidate_preprocessors(self, model_name, params):
        candidates = ["none"]
        if model_name in {
            "k_nearest_neighbors",
            "liblinear_svr",
            "libsvm_svr",
            "mlp",
            "gaussian_process",
            "ard_regression",
            "sgd",
        }:
            candidates.extend(["standard", "robust"])
        return candidates

    def convert_score_to_cost(self, score):
        return float(-score)

    def return_failure_score(self):
        return -1e12 if self.scoring != "r2" else float("-inf")

    def get_cv_splitter(self, cv_num_of_splits):
        return KFold(
            n_splits=cv_num_of_splits, shuffle=True, random_state=self.random_state
        )

    def make_subsample_splitter(self, target_size):
        return ShuffleSplit(
            n_splits=1, train_size=target_size, random_state=self.random_state
        )

    def resolve_cv_folds(self, y, requested_folds):
        sample_count = len(np.asarray(y).reshape(-1))
        return max(2, min(requested_folds, sample_count))
