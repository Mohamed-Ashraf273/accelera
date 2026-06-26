import warnings
from abc import ABC
from abc import abstractmethod

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_X_y


class BaseAutoML(BaseEstimator, ABC):
    def __init__(
        self,
        time_budget=None,
        n_trials=50,
        scoring=None,
        random_state=None,
        use_ensemble=True,
        per_run_time_limit=None,
        ensemble_voting_size=10,
        ensemble_strategy="stacked",
        cv=5,
        stacked_base_size=4,
        stacked_include_original_features_in_meta=False,
        stack_n_jobs=-1,
        stacked_bagging_n_estimators=5,
        inner_n_jobs=1,
        disable_evaluation_timeout=False,
        sample_size_from_config_space=256,
        num_of_trial_to_try_in_parallel=3,
        n_initial_points=5,
        verbose=1,
    ):
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.scoring = scoring
        self.random = random_state
        self.use_ensemble = use_ensemble
        self.per_run_time_limit = per_run_time_limit
        self.ensemble_voting_size = ensemble_voting_size
        self.ensemble_strategy = ensemble_strategy
        self.cv = cv
        self.stacked_base_size = stacked_base_size
        self.stacked_include_original_features_in_meta = (
            stacked_include_original_features_in_meta
        )
        self.stack_n_jobs = stack_n_jobs
        self.stacked_bagging_n_estimators = stacked_bagging_n_estimators
        self.inner_n_jobs = inner_n_jobs
        self.disable_evaluation_timeout = disable_evaluation_timeout
        self.sample_size_from_config_space = sample_size_from_config_space
        self.num_of_trial_to_try_in_parallel = num_of_trial_to_try_in_parallel
        self.n_initial_points = n_initial_points
        self.verbose = verbose

        self.reset()

    def reset(self):
        self.leaderboard = None
        self.final_model = None
        self.best_score = None
        self.is_fitted = False

    @abstractmethod
    def get_default_scoring(self):
        "return default scoring metric depend on the task"

    def get_effective_scoring(self):
        if self.scoring is not None:
            return self.scoring
        return self.get_default_scoring()

    @abstractmethod
    def build_engine(self):
        "build the search engine depend on the task"

    def validate_train_data(self, X, y):
        X_valid, y_valid = check_X_y(X, y, accept_sparse=True)
        return X_valid, y_valid

    def validate_test_data(self, X):
        return check_array(X, accept_sparse=True)

    def fit(self, X, y):
        self.reset()
        X_valid, y_valid = self.validate_train_data(X, y)
        engine = self.build_engine()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = engine.search(X_valid, y_valid)
        self.leaderboard = result.leaderboard if result is not None else None
        self.final_model = result.final_model if result is not None else None
        self.best_score = result.best_score if result is not None else None
        self.is_fitted = self.final_model is not None

        if not self.is_fitted:
            raise RuntimeError("no valid fitted model found")

        return self

    def get_final_model(self):
        if not self.is_fitted:
            raise RuntimeError("call fit(X,y) before using this estimator.")

        return self.final_model

    def predict(self, X):
        model = self.get_final_model()
        X_valid = self.validate_test_data(X)
        return model.predict(X_valid)

    def predict_proba(self, X):
        model = self.get_final_model()
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                "The final model does not implement `predict_proba`."
            )
        X_valid = self.validate_test_data(X)
        return model.predict_proba(X_valid)

    def return_leaderboard(self, top_n=5):
        if not self.is_fitted:
            raise RuntimeError("call fit(X,y) before using this estimator.")

        if self.leaderboard is None:
            return self.leaderboard

        return self.leaderboard[:top_n]
