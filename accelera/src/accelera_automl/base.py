from abc import ABC
from abc import abstractmethod
from typing import Mapping

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_X_y


class BaseAutoML(BaseEstimator, ABC):
    def __init__(
        self,
        *,
        time_budget=None,
        per_run_time_limit=None,
        n_trials=50,
        cv=5,
        scoring=None,
        random_state=None,
        use_ensemble=True,
        ensemble_voting_size=10,
        ensemble_strategy="stacked",
        stacked_base_size=4,
        stacked_bagging_n_estimators=5,
        stacked_include_original_features_in_meta=False,
        n_jobs=None,
        search_n_parallel=3,
        stack_n_jobs=None,
        inner_n_jobs=1,
        verbose=1,
        disable_evaluation_timeout=False,
        candidate_pool_size=256,
        n_initial_points=5,
    ):

        self.time_budget = time_budget
        self.per_run_time_limit = per_run_time_limit
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.use_ensemble = use_ensemble
        self.ensemble_voting_size = ensemble_voting_size
        self.ensemble_strategy = ensemble_strategy
        self.stacked_base_size = stacked_base_size
        self.stacked_bagging_n_estimators = stacked_bagging_n_estimators
        self.stacked_include_original_features_in_meta = (
            stacked_include_original_features_in_meta
        )
        self.n_jobs = n_jobs
        self.search_n_parallel = search_n_parallel
        self.stack_n_jobs = stack_n_jobs
        self.inner_n_jobs = inner_n_jobs
        self.verbose = verbose
        self.disable_evaluation_timeout = disable_evaluation_timeout
        self.candidate_pool_size = candidate_pool_size
        self.n_initial_points = n_initial_points

        self.reset()

    @abstractmethod
    def get_default_scoring(self):
        "return default scoring metric"

    @abstractmethod
    def build_engine(self):
        "build the search engine"

    def reset(self):
        self.best_model = None
        self.best_score = None
        self.runhistory = None
        self.leaderboard = None
        self.ensemble = None
        self.final_model = None
        self.is_fitted = False

    def get_effective_scoring(self):
        if self.scoring is not None:
            return self.scoring
        return self.get_default_scoring()

    def validate_fit_inputs(self, X, y):
        X_valid, y_copy = check_X_y(X, y, accept_sparse=True)
        return X_valid, y_copy

    def validate_predict_input(self, X):
        return check_array(X, accept_sparse=True)

    def apply_fit_result(self, result):
        self.best_model = self.read_result_value(result, "best_model")
        self.best_score = self.read_result_value(result, "best_score")
        self.runhistory = self.read_result_value(result, "runhistory")
        self.leaderboard = self.read_result_value(result, "leaderboard")
        self.ensemble = self.read_result_value(result, "ensemble")
        final_model = self.read_result_value(result, "final_model")

        if final_model is None:
            final_model = (
                self.ensemble if self.ensemble is not None else self.best_model
            )

        self.final_model = final_model
        self.is_fitted = self.final_model is not None

        if not self.is_fitted:
            raise RuntimeError("no valid fitted model found")

    @staticmethod
    def read_result_value(result, name):
        if result is None:
            return None

        if isinstance(result, Mapping):
            return result.get(name, None)

        if hasattr(result, name):
            return getattr(result, name)

        return None

    def get_final_model(self):
        if not self.is_fitted:
            raise RuntimeError("call fit(X,y) before using this estimator.")

        return self.final_model

    def fit(self, X, y):
        self.reset()
        X_valid, y_valid = self.validate_fit_inputs(X, y)
        engine = self.build_engine()
        result = engine.search(X_valid, y_valid)
        self.apply_fit_result(result)
        return self

    def predict(self, X):
        model = self.get_final_model()
        X_valid = self.validate_predict_input(X)
        return model.predict(X_valid)

    def predict_proba(self, X):
        model = self.get_final_model()
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                "The final model does not implement `predict_proba`."
            )
        X_valid = self.validate_predict_input(X)
        return model.predict_proba(X_valid)

    def return_leaderboard(self, top_n=None):
        if not self.is_fitted:
            raise RuntimeError("call fit(X,y) before using this estimator.")

        if self.leaderboard is None or top_n is None:
            return self.leaderboard

        return self.leaderboard[:top_n]

    def get_runhistory(self):
        if not self.is_fitted:
            raise RuntimeError("call fit(X,y) before using this estimator.")

        return self.runhistory
