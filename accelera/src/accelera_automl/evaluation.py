from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from .components import get_classification_components
from .components import get_regression_components
from .configspace_search_space import configuration_space_to_dict


@dataclass
class EvaluationResult:
    model_name: str | None = None
    params: dict | None = None
    preprocessing: str | None = None
    score: float = 0.0
    cost: float = 0.0
    duration: float = 0.0
    status: str | None = None
    error: str | None = None
    evaluation_level_stage: int = 0
    sample_fraction: float = 1.0
    cv_folds: int = 5
    model_budget: float = 1.0


@dataclass
class TrialSpecs:
    stage: int = 0
    sample_fraction: float = 1.0
    cv_folds: int | None = None
    model_budget: float = 1.0


class BaseModelEvaluator:
    def __init__(
        self,
        *,
        cv=5,
        scoring="accuracy",
        random_state=None,
        n_jobs=None,
        per_run_time_limit=None,
    ):
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.per_run_time_limit = per_run_time_limit

    def evaluate(self, config, X, y, evaluation_level=None):

        started_at = perf_counter()
        config_dict = configuration_space_to_dict(config)
        model_name = config_dict["model_name"]
        params = config_dict["params"]
        effective_evaluation_level = self.normalize_evaluation_level(
            evaluation_level
        )
        X_eval, y_eval = self.get_sample(
            X, y, effective_evaluation_level.sample_fraction
        )

        try:
            preprocessing_name, score = self.evaluate_different_preprocessing(
                model_name,
                params,
                X_eval,
                y_eval,
                effective_evaluation_level=effective_evaluation_level,
            )
            status = "success"
            error = None

        except Exception as exc:
            preprocessing_name = "none"
            score = self.return_failure_score()
            status = "failed"
            error = str(exc)

        duration = float(perf_counter() - started_at)
        cost = self.convert_score_to_cost(score)
        return EvaluationResult(
            model_name=model_name,
            params=params,
            preprocessing=preprocessing_name,
            score=score,
            cost=cost,
            duration=duration,
            status=status,
            error=error,
            evaluation_level_stage=effective_evaluation_level.stage,
            sample_fraction=effective_evaluation_level.sample_fraction,
            cv_folds=effective_evaluation_level.cv_folds or self.cv,
            model_budget=effective_evaluation_level.model_budget,
        )

    def score_config(
        self,
        config,
        X,
        y,
    ):
        return self.evaluate(config, X, y).cost

    def build_model(
        self,
        model_name,
        params,
        preprocessing="none",
        evaluation_level=None,
    ):
        estimator = self.build_estimator(
            model_name, params, evaluation_level=evaluation_level
        )
        transformer = self.build_preprocessor(preprocessing)
        if transformer is None:
            return estimator
        return Pipeline(
            steps=[
                ("preprocessor", transformer),
                ("estimator", estimator),
            ]
        )

    # Try different preprocessing based on the model.
    def evaluate_different_preprocessing(
        self,
        model_name,
        params,
        X,
        y,
        effective_evaluation_level,
    ):
        n_splits = self.resolve_cv_folds(
            y, effective_evaluation_level.cv_folds or self.cv
        )
        splitter = self.make_cv_splitter(n_splits)

        best_preprocessing = "none"
        best_score = float("-inf")

        for preprocessing in self.get_candidate_preprocessors(model_name, params):
            model = self.build_model(
                model_name,
                params,
                preprocessing=preprocessing,
                evaluation_level=effective_evaluation_level,
            )
            scores = cross_val_score(
                model,
                X,
                y,
                cv=splitter,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
            score = float(scores.mean())
            if score > best_score:
                best_score = score
                best_preprocessing = preprocessing

        return best_preprocessing, best_score

    def build_preprocessor(self, name):
        if name == "none":
            return None
        if name == "standard":
            return StandardScaler()
        if name == "robust":
            return RobustScaler()
        if name == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unsupported preprocessing strategy `{name}`.")

    def normalize_evaluation_level(self, evaluation_level=None):

        if evaluation_level is None:  # try at max level.
            return TrialSpecs(
                stage=0, sample_fraction=1.0, cv_folds=self.cv, model_budget=1.0
            )

        return TrialSpecs(  # try at specified level.
            stage=int(evaluation_level.stage),
            sample_fraction=float(evaluation_level.sample_fraction),
            cv_folds=int(evaluation_level.cv_folds or self.cv),
            model_budget=float(evaluation_level.model_budget),
        )

    def get_sample(
        self,
        X,
        y,
        sample_fraction,
    ):
        if sample_fraction == 1.0:
            return X, y

        num_of_samples = len(y)
        target_size = round(num_of_samples * sample_fraction)
        splitter = self.make_subsample_splitter(target_size)

        indices, _ = next(splitter.split(np.zeros(num_of_samples), y))
        return X[indices], y[indices]

    def apply_model_budget(
        self,
        estimator,
        model_name,
        params,
        evaluation_level,
    ):
        if evaluation_level.model_budget == 1.0:
            return

        budget = {
            "random_forest": ("n_estimators", 16),  # (param,min value)
            "extra_trees": ("n_estimators", 16),
            "gradient_boosting": ("n_estimators", 16),
            "lightgbm": ("n_estimators", 16),
            "xgboost": ("n_estimators", 16),
            "catboost": ("iterations", 16),
            "hist_gradient_boosting": ("max_iter", 16),
            "adaboost": ("n_estimators", 8),
            "logistic_regression": ("max_iter", 50),
            "mlp": ("max_iter", 30),
            "gaussian_process": ("max_iter_predict", 10),
        }
        budget_param = budget.get(model_name)
        if budget_param is None:
            return

        param, minimum = budget_param
        base_value = params.get(param)
        if base_value is None:
            return

        scaled_value = max(
            minimum, int(round(float(base_value) * evaluation_level.model_budget))
        )
        if hasattr(estimator, "set_params"):
            try:
                estimator.set_params(**{param: scaled_value})
            except ValueError:
                pass

    def make_cv_splitter(self, n_splits):
        raise NotImplementedError

    def make_subsample_splitter(self, target_size):
        raise NotImplementedError

    def resolve_cv_folds(self, y, requested_folds):
        raise NotImplementedError

    def build_estimator(self, model_name, params, evaluation_level):
        raise NotImplementedError

    def get_candidate_preprocessors(self, model_name, params):
        raise NotImplementedError

    def convert_score_to_cost(self, score):
        raise NotImplementedError

    def return_failure_score(self):
        raise NotImplementedError


class ClassificationEvaluator(BaseModelEvaluator):
    def __init__(
        self,
        *,
        cv=5,
        scoring="accuracy",
        random_state=None,
        n_jobs=None,
        balance_classes=False,
        per_run_time_limit=None,
    ):
        super().__init__(
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs,
            per_run_time_limit=per_run_time_limit,
        )
        self.balance_classes = balance_classes

    def build_estimator(
        self,
        model_name,
        params,
        evaluation_level=None,
    ):
        components = get_classification_components()
        if model_name not in components:
            raise ValueError(f"unsupported classification model `{model_name}`.")
        component = components[model_name]
        estimator = component.build_estimator(
            params,
            self.random_state,
            self.n_jobs,
            self.balance_classes,
        )
        self.apply_model_budget(
            estimator,
            model_name,
            params,
            self.normalize_evaluation_level(evaluation_level),
        )
        return estimator

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

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(candidates))

    def convert_score_to_cost(self, score):
        if self.scoring == "accuracy":
            return float(1.0 - score)
        return float(-score)

    def make_cv_splitter(self, n_splits):
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
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
        # Each fold must contain at least one sample from every class.
        max_supported = int(np.min(counts))
        return max(2, min(int(requested_folds), max_supported))


class RegressionEvaluator(BaseModelEvaluator):
    def __init__(
        self,
        *,
        cv: int = 5,
        scoring: str = "r2",
        random_state: int | None = None,
        n_jobs: int | None = None,
        per_run_time_limit: float | None = None,
    ) -> None:
        super().__init__(
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs,
            per_run_time_limit=per_run_time_limit,
        )

    def build_estimator(self, model_name, params, evaluation_level):
        components = get_regression_components()
        if model_name not in components:
            raise ValueError(f"unsupported regression model `{model_name}`.")
        component = components[model_name]
        estimator = component.build_estimator(params, self.random_state, self.n_jobs)
        self.apply_model_budget(
            estimator,
            model_name,
            params,
            self.normalize_evaluation_level(evaluation_level),
        )
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
        return list(dict.fromkeys(candidates))

    def convert_score_to_cost(self, score):
        return float(-score)

    def return_failure_score(self):
        return -1e12 if self.scoring != "r2" else float("-inf")

    def make_cv_splitter(self, n_splits):
        return KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    def make_subsample_splitter(self, target_size):
        return ShuffleSplit(
            n_splits=1, train_size=target_size, random_state=self.random_state
        )

    def resolve_cv_folds(self, y, requested_folds) -> int:
        sample_count = len(np.asarray(y).reshape(-1))
        return max(2, min(int(requested_folds), sample_count))
