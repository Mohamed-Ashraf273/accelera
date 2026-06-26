import numpy as np
from sklearn.metrics import r2_score

from accelera.src.accelera_automl.base import BaseAutoML
from accelera.src.accelera_automl.core.automl import AutoMLEngine


class AutoMLRegressor(BaseAutoML):
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
        search_n_parallel=3,
        stack_n_jobs=-1,
        n_jobs=None,
        inner_n_jobs=1,
        disable_evaluation_timeout=False,
        allowed_models=None,
        use_meta_learning=True,
        meta_learning_top_datasets=5,
        meta_learning_top_configs_per_dataset=3,
        max_meta_learning_warmstarts=10,
        candidate_pool_size=256,
        n_initial_points=5,
        verbose=1,
    ):
        super().__init__(
            time_budget=time_budget,
            n_trials=n_trials,
            cv=cv,
            use_ensemble=use_ensemble,
            scoring=scoring,
            per_run_time_limit=per_run_time_limit,
            random_state=random_state,
            ensemble_strategy=ensemble_strategy,
            stacked_base_size=stacked_base_size,
            stacked_include_original_features_in_meta=stacked_include_original_features_in_meta,
            num_of_trial_to_try_in_parallel=search_n_parallel,
            stacked_bagging_n_estimators=stacked_bagging_n_estimators,
            stack_n_jobs=n_jobs if n_jobs is not None else stack_n_jobs,
            ensemble_voting_size=ensemble_voting_size,
            inner_n_jobs=inner_n_jobs,
            disable_evaluation_timeout=disable_evaluation_timeout,
            sample_size_from_config_space=candidate_pool_size,
            n_initial_points=n_initial_points,
            verbose=verbose,
        )

        self.allowed_models = allowed_models
        self.use_meta_learning = use_meta_learning
        self.meta_learning_top_datasets = meta_learning_top_datasets
        self.meta_learning_top_configs_per_dataset = (
            meta_learning_top_configs_per_dataset
        )
        self.max_meta_learning_warmstarts = max_meta_learning_warmstarts

    def get_default_scoring(self):
        return "r2"

    def build_engine(self):
        return AutoMLEngine(
            task="regression",
            time_budget=self.time_budget,
            per_run_time_limit=self.per_run_time_limit,
            n_trials=self.n_trials,
            cv=self.cv,
            scoring=self.get_effective_scoring(),
            random_state=self.random,
            use_ensemble=self.use_ensemble,
            ensemble_voting_size=self.ensemble_voting_size,
            ensemble_strategy=self.ensemble_strategy,
            stacked_base_size=self.stacked_base_size,
            stacked_bagging_n_estimators=self.stacked_bagging_n_estimators,
            stacked_include_original_features_in_meta=self.stacked_include_original_features_in_meta,
            search_n_parallel=self.num_of_trial_to_try_in_parallel,
            stack_n_jobs=self.stack_n_jobs,
            inner_n_jobs=self.inner_n_jobs,
            disable_evaluation_timeout=self.disable_evaluation_timeout,
            allowed_models=self.allowed_models,
            use_meta_learning=self.use_meta_learning,
            meta_learning_top_datasets=self.meta_learning_top_datasets,
            meta_learning_top_configs_per_dataset=self.meta_learning_top_configs_per_dataset,
            max_meta_learning_warmstarts=self.max_meta_learning_warmstarts,
            candidate_pool_size=self.sample_size_from_config_space,
            n_initial_points=self.n_initial_points,
            verbose=self.verbose,
        )

    def fit(self, X, y):
        self.n_outputs = (
            1 if np.asarray(y).ndim == 1 else int(np.asarray(y).shape[1])
        )
        return super().fit(X, y)

    def score(self, X, y):
        predictions = self.predict(X)
        return float(r2_score(y, predictions))
