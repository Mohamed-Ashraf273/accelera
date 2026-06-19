
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from ..configspace_search_space import (
    configuration_space_to_dict,
    return_classification_config_space,
    return_regression_config_space,
)
from ..evaluation import ClassificationEvaluator, TrialSpecs, RegressionEvaluator
from ..meta_learning import (
    compute_basic_classification_metafeatures,
    compute_basic_regression_metafeatures,
    get_meta_learning_warmstarts,
)
from ..optimization.smac_optimizer import Optimizer
from ..stacked_ensemble import (
    StackedEnsembleClassifier,
)
from ..stacked_ensemble_regression import (
    StackedEnsembleRegressor,
)


class AutoMLResult:
    def __init__(self, best_model, best_score, leaderboard, ensemble, final_model, runhistory=None):
        self.best_model = best_model
        self.best_score = best_score
        self.leaderboard = leaderboard
        self.ensemble = ensemble
        self.final_model = final_model
        self.runhistory = runhistory


class AutoMLEngine:
    DEFAULT_FIDELITY_STAGES = (
        TrialSpecs(stage=0, sample_fraction=0.2, cv_folds=2, model_budget=0.25),
        TrialSpecs(stage=1, sample_fraction=0.5, cv_folds=3, model_budget=0.6),
        TrialSpecs(stage=2, sample_fraction=1.0, cv_folds=None, model_budget=1.0),
    )

    def __init__(
        self,
        *,
        task,
        time_budget = None,
        per_run_time_limit = None,
        n_trials = 50,
        cv = 5,
        scoring = None,
        random_state = None,
        use_ensemble = True,
        ensemble_voting_size = 10,
        ensemble_strategy = "stacked",
        stacked_base_size = 4,
        stacked_bagging_n_estimators = 5,
        stacked_include_original_features_in_meta = False,
        n_jobs = None,
        search_n_parallel = 3,
        stack_n_jobs = None,
        inner_n_jobs = 1,
        verbose = 1,
        disable_evaluation_timeout = False,
        balance_classes = False,
        allowed_models = None,
        use_meta_learning = True,
        meta_learning_top_datasets = 5,
        meta_learning_top_configs_per_dataset = 3,
        max_meta_learning_warmstarts = 10,
        candidate_pool_size = 256,
        n_initial_points = 5,
    ):
        self.task = task
        self.time_budget = time_budget
        self.per_run_time_limit = per_run_time_limit
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.ensemble = use_ensemble
        self.ensemble_voting_size = ensemble_voting_size
        self.ensemble_strategy = ensemble_strategy
        self.stacked_base_size = stacked_base_size
        self.stacked_bagging_n_estimators = stacked_bagging_n_estimators
        self.stacked_include_original_features_in_meta = stacked_include_original_features_in_meta
        self.n_jobs = n_jobs
        self.search_n_parallel = max(1, int(search_n_parallel))
        self.stack_n_jobs = n_jobs if stack_n_jobs is None else stack_n_jobs
        self.inner_n_jobs = inner_n_jobs
        self.verbose = verbose
        self.disable_evaluation_timeout = disable_evaluation_timeout
        self.balance_classes = balance_classes
        self.allowed_models = allowed_models
        self.use_meta_learning = use_meta_learning
        self.meta_learning_top_datasets = meta_learning_top_datasets
        self.meta_learning_top_configs_per_dataset = meta_learning_top_configs_per_dataset
        self.max_meta_learning_warmstarts = max_meta_learning_warmstarts
        self.candidate_pool_size = candidate_pool_size
        self.n_initial_points = n_initial_points

    def search(self, X, y):

        models = self.resolve_allowed_models(X)
        configspace = self.build_configspace(models)
        warmstart_configs = self.get_meta_learning_warmstarts(configspace, X, y)
        evaluator = self.build_evaluator()
        optimizer = self.build_optimizer(configspace, evaluator, X, y, warmstart_configs)

        optimization_result = optimizer.optimize()
        best_config = optimization_result.best_config
        if best_config is None:
            raise RuntimeError("optimizer did not return a best configuration.")

        best_config_result = optimization_result.best_config_result
        leaderboard = self.build_leaderboard(optimization_result.runhistory)

        best_model = self.fit_final_model(
            best_config,
            evaluator,
            X,
            y,
            preprocessing=best_config_result.preprocessing,
        )
        ensemble_model, ensemble_score = self.build_ensemble(optimization_result.runhistory, evaluator, X, y)
        best_model_score = float(best_config_result.score)
        final_model = best_model
        final_score = best_model_score

        if ensemble_model is not None and ensemble_score is not None:
            leaderboard = self.append_ensemble_to_leaderboard(
                leaderboard=leaderboard,
                ensemble_score=ensemble_score,
            )
            if ensemble_score >= best_model_score:
                final_model = ensemble_model
                final_score = ensemble_score

        return AutoMLResult(
            best_model=best_model,
            best_score=float(final_score),
            leaderboard=leaderboard,
            ensemble=ensemble_model,
            final_model=final_model,
            runhistory=optimization_result.runhistory,
        )

    def build_configspace(self, models):
        if self.task == "classification":
            return return_classification_config_space(allowed_models=models)
        return return_regression_config_space(allowed_models=models)

    def build_evaluator(self):
        if self.task == "classification":
            return ClassificationEvaluator(
                cv=self.cv,
                scoring=self.scoring,
                random_state=self.random_state,
                n_jobs=self.inner_n_jobs,
                balance_classes=self.balance_classes,
                per_run_time_limit=self.resolve_per_run_time_limit(),
            )
        return RegressionEvaluator(
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=self.inner_n_jobs,
            per_run_time_limit=self.resolve_per_run_time_limit(),
        )

    def build_optimizer(
        self,
        configspace,
        evaluator,
        X,
        y,
        initial_configurations,
    ):
        return Optimizer(
            configspace=configspace,
            evaluator=evaluator,
            X=X,
            y=y,
            n_trials=self.n_trials,
            time_budget=self.time_budget,
            random_state=self.random_state,
            per_run_time_limit=self.resolve_per_run_time_limit(),
            initial_configurations=initial_configurations,
            n_parallel=self.search_n_parallel,
            verbose=self.verbose,
            evaluation_level=list(self.DEFAULT_FIDELITY_STAGES),
            candidate_pool_size=self.candidate_pool_size,
            n_initial_points=self.n_initial_points,
        )

    def fit_final_model(
        self,
        best_config,
        evaluator,
        X,
        y,
        preprocessing = "none",
    ):
        config_dict = configuration_space_to_dict(best_config)
        model = evaluator.build_model(
            config_dict["model_name"],
            config_dict["params"],
            preprocessing=preprocessing,
        )
        model.fit(X, y)
        return model

  
    def build_ensemble(
        self,
        results,
        evaluator,
        X,
        y,
    ):
        if not self.ensemble:
            return None, None

        effective_ensemble_strategy = self.resolve_ensemble_strategy(X)

        if effective_ensemble_strategy == "stacked":
            return self.build_stacked_ensemble(results, evaluator, X, y)
        if effective_ensemble_strategy != "voting":
            raise ValueError(f"unsupported ensemble strategy `{effective_ensemble_strategy}`.")

        if self.task == "regression":
            return self.build_regression_voting_ensemble(results, evaluator, X, y)

        ranked_rows = [
            row
            for row in sorted(
                results,
                key=lambda item: item.get("score", float("-inf")),
                reverse=True,
            )
            if row.get("status") == "success" and row.get("config") is not None
        ]

        if len(ranked_rows) < 2:
            return None, None

        candidate_estimators = []
        seen_signatures = set()
        for row in ranked_rows:
            config_dict = configuration_space_to_dict(row["config"])
            signature = (
                config_dict["model_name"],
                tuple(sorted(config_dict["params"].items())),
                row.get("preprocessing", "none"),
            )
            if signature in seen_signatures:
                continue

            estimator = evaluator.build_model(
                config_dict["model_name"],
                config_dict["params"],
                preprocessing=row.get("preprocessing", "none"),
            )
            estimator_name = f"{config_dict['model_name']}_{len(candidate_estimators)}"
            candidate_estimators.append((estimator_name, estimator))
            seen_signatures.add(signature)

            if len(candidate_estimators) == self.ensemble_voting_size * 3:
                break

        if len(candidate_estimators) < 2:
            return None, None

        selected_estimators = []
        remaining_candidates = list(candidate_estimators)
        best_ensemble_score = float("-inf")

        while remaining_candidates and len(selected_estimators) < self.ensemble_voting_size:
            best_candidate_idx = None
            best_candidate_score = best_ensemble_score

            for idx, (candidate_name, candidate_estimator) in enumerate(remaining_candidates):
                trial_estimators = selected_estimators + [(candidate_name, candidate_estimator)]
                voting = self.get_voting_strategy(trial_estimators)
                trial_ensemble = VotingClassifier(
                    estimators=trial_estimators,
                    voting=voting,
                    n_jobs=self.inner_n_jobs,
                )
                trial_score = self.evaluate_estimator(trial_ensemble, X, y)
                if trial_score > best_candidate_score:
                    best_candidate_score = trial_score
                    best_candidate_idx = idx

            if best_candidate_idx is None:
                break

            candidate_estimator = remaining_candidates.pop(best_candidate_idx)
            selected_estimators.append(candidate_estimator)
            best_ensemble_score = best_candidate_score

        if len(selected_estimators) < 2:
            return None, None

        ensemble_model = VotingClassifier(
            estimators=selected_estimators,
            voting=self.get_voting_strategy(selected_estimators),
            n_jobs=self.inner_n_jobs,
        )
        ensemble_model.fit(X, y)
        return ensemble_model, float(best_ensemble_score)

    def build_regression_voting_ensemble(
        self,
        results,
        evaluator,
        X,
        y,
    ):
        ranked_rows = [
            row
            for row in sorted(results, key=lambda item: item.get("score", float("-inf")), reverse=True)
            if row.get("status") == "success" and row.get("config") is not None
        ]
        if len(ranked_rows) < 2:
            return None, None

        candidate_estimators = []
        seen_signatures = set()
        for row in ranked_rows:
            config_dict = configuration_space_to_dict(row["config"])
            signature = (
                config_dict["model_name"],
                tuple(sorted(config_dict["params"].items())),
                row.get("preprocessing", "none"),
            )
            if signature in seen_signatures:
                continue
            estimator = evaluator.build_model(
                config_dict["model_name"],
                config_dict["params"],
                preprocessing=row.get("preprocessing", "none"),
            )
            candidate_estimators.append((f"{config_dict['model_name']}_{len(candidate_estimators)}", estimator))
            seen_signatures.add(signature)
            if len(candidate_estimators) == 3 * self.ensemble_voting_size:
                break

        if len(candidate_estimators) < 2:
            return None, None

        selected_estimators = []
        best_ensemble_score = float("-inf")
        while candidate_estimators and len(selected_estimators) < self.ensemble_voting_size:
            best_candidate_idx = None
            best_candidate_score = best_ensemble_score
            for idx, (candidate_name, candidate_estimator) in enumerate(candidate_estimators):
                trial_ensemble = VotingRegressor(estimators=selected_estimators + [(candidate_name, candidate_estimator)])
                trial_score = self.evaluate_estimator(trial_ensemble, X, y)
                if trial_score > best_candidate_score:
                    best_candidate_score = trial_score
                    best_candidate_idx = idx
            if best_candidate_idx is None:
                break
            candidate_name, candidate_estimator = candidate_estimators.pop(best_candidate_idx)
            selected_estimators.append((candidate_name, candidate_estimator))
            best_ensemble_score = best_candidate_score

        if len(selected_estimators) < 2:
            return None, None
        
        ensemble_model = VotingRegressor(estimators=selected_estimators)
        ensemble_model.fit(X, y)
        return ensemble_model, float(best_ensemble_score)

    def resolve_ensemble_strategy(self, X):
        n_rows = len(X)
        if self.ensemble_strategy == "stacked" and n_rows > 50_000:
            if self.verbose:
                print(
                    "[AutoML] Large dataset detected; switching ensemble strategy "
                    "from stacked to voting."
                )
            return "voting"
        return self.ensemble_strategy

    def build_stacked_ensemble(
        self,
        cv_results,
        evaluator,
        X,
        y,
    ):
        ranked_rows = [
            row
            for row in sorted(
                cv_results,
                key=lambda item: item.get("score", float("-inf")),
                reverse=True,
            )
            if row.get("status") == "success" and row.get("config") is not None
        ]
        if len(ranked_rows) < 2:
            return None, None

        base_estimators = self.select_diverse_base_estimators(ranked_rows, evaluator)
        if len(base_estimators) < 3:
            return None, None

        meta_estimators = self.select_meta_estimators()
        if len(meta_estimators) < 1:
            return None, None

        if self.task == "regression":
            ensemble_model = StackedEnsembleRegressor(
                base_estimators=base_estimators,
                meta_estimators=meta_estimators,
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=self.stack_n_jobs,
                inner_n_jobs=self.inner_n_jobs,
                bagging_n_estimators=self.stacked_bagging_n_estimators,
                include_original_features_in_meta=self.stacked_include_original_features_in_meta,
                scoring=self.scoring,
                verbose=self.verbose,
            )
            ensemble_model.fit(X, y)
            return ensemble_model, float(ensemble_model.forward_selection_.score)

        ensemble_model = StackedEnsembleClassifier(
            base_estimators=base_estimators,
            meta_estimators=meta_estimators,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=self.stack_n_jobs,
            inner_n_jobs=self.inner_n_jobs,
            bagging_n_estimators=self.stacked_bagging_n_estimators,
            include_original_features_in_meta=self.stacked_include_original_features_in_meta,
            scoring=self.scoring,
            verbose=self.verbose,
        )
        ensemble_model.fit(X, y)
        return ensemble_model, float(ensemble_model.forward_selection_.score)

    def select_meta_estimators(
        self,
    ):
        if self.task == "classification":
           return [
                (
                    "meta_logistic_regression",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        penalty="l2",
                        max_iter=1000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        else:
            return [
        (
            "meta_ridge_regressor",
            Ridge(alpha=1.0),
        ),
    ]

    def select_diverse_base_estimators(
        self,
        ranked_rows,
        evaluator,
    ):
        candidate_rows = []
        seen_signatures = set()
        num_of_candidates = 10

        for row in ranked_rows:
            config_dict = configuration_space_to_dict(row["config"])
            family_name = config_dict["model_name"]
            signature = (
                family_name,
                tuple(sorted(config_dict["params"].items())),
                row.get("preprocessing", "none"),
            )
            if signature in seen_signatures:
                continue

            estimator = evaluator.build_model(
                config_dict["model_name"],
                config_dict["params"],
                preprocessing=row.get("preprocessing", "none"),
            )
            estimator_name = f"base_{family_name}_{len(candidate_rows)}"
            candidate_rows.append((estimator_name, estimator))
            seen_signatures.add(signature)

            if len(candidate_rows) >= num_of_candidates:
                break

        return [(name, estimator) for name, estimator in candidate_rows]



    






    def evaluate_estimator(self, estimator, X, y):
        if self.task == "classification":
            splitter = StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            splitter = KFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
        scores = cross_val_score(
            clone(estimator),
            X,
            y,
            cv=splitter,
            scoring=self.scoring,
            n_jobs=self.inner_n_jobs,
        )
        return float(scores.mean())

    @staticmethod
    def build_leaderboard(result):
        ranked = sorted(
            result,
            key=lambda row: row.get("score", float("-inf")),
            reverse=True,
        )

        leaderboard = []
        for rank, row in enumerate(ranked, start=1):
            leaderboard.append(
                {
                    "rank": rank,
                    "trial_id": row.get("trial_id", rank - 1),
                    "model_name": row["model_name"],
                    "preprocessing": row.get("preprocessing", "none"),
                    "cv_score": row["score"],
                    "error": row.get("error"),
                    "params": row["params"],
                }
            )

        return leaderboard

    def append_ensemble_to_leaderboard(
        self,
        *,
        leaderboard,
        ensemble_score,
    ):
        updated = list(leaderboard)
        updated.append(
            {
                "rank": None,
                "trial_id": "ensemble",
                "model_name": f"{self.ensemble_strategy}_ensemble",
                "preprocessing": "stacked",
                "cv_score": float(ensemble_score),
                "error": None,
                "params": {
                    "stacked_base_size": self.stacked_base_size,
                    "stacked_bagging_n_estimators": self.stacked_bagging_n_estimators,
                    "include_original_features_in_meta": self.stacked_include_original_features_in_meta,
                },
            }
        )
        reranked = sorted(
            updated,
            key=lambda row: row.get("cv_score", float("-inf")),
            reverse=True,
        )
        for rank, row in enumerate(reranked, start=1):
            row["rank"] = rank
        return reranked

    def resolve_allowed_models(self, X):

        candidate_models = self.allowed_models
        disabled_models = []
        n_samples = len(X) 

        if self.task == "classification":
            if n_samples >= 10_000:
                disabled_models.append("knn")
                disabled_models.append("gaussian_process")
            if n_samples >= 50_000:
                disabled_models.append("svc")
                disabled_models.append("mlp")
            if self.features_have_negative_values(X):
                disabled_models.append("multinomial_nb")
        else:
            if n_samples >= 10_000:
                disabled_models.append("gaussian_process")
            if n_samples >= 50_000:
                disabled_models.append("libsvm_svr")
                disabled_models.append("mlp")

        if not disabled_models:
            return candidate_models

        disabled_set = set(disabled_models)
        if candidate_models is None:
            configspace = (
                return_classification_config_space()
                if self.task == "classification"
                else return_regression_config_space()
            )
            filtered_models = [
                model_name
                for model_name in configspace["model_name"].choices
                if model_name not in disabled_set
            ]
        else:
            filtered_models = [
                model_name for model_name in candidate_models if model_name not in disabled_set
            ]

        if self.verbose:
            print(
                "dataset-aware filtering disabled models:",
                disabled_set
            )
        return filtered_models

    @staticmethod
    def features_have_negative_values(X):
        try:
            if hasattr(X, "select_dtypes"):
                numeric_frame = X.select_dtypes(include=["number"])
                if numeric_frame.shape[1] == 0:
                    return False
                numeric_values = numeric_frame.to_numpy()
            else:
                numeric_values = np.asarray(X)
                if numeric_values.dtype.kind not in {"i", "u", "f", "b"}:
                    return False
            return bool(np.nanmin(numeric_values) < 0)
        except Exception:
            return False

    @staticmethod
    def get_voting_strategy(estimators):
        if all(hasattr(estimator[1], "predict_proba") for estimator in estimators):
            return "soft"
        return "hard"

    def get_meta_learning_warmstarts(self, configspace, X, y):
        if not self.use_meta_learning:
            return []

        try:
            metafeatures = (
                compute_basic_classification_metafeatures(X, y)
                if self.task == "classification"
                else compute_basic_regression_metafeatures(X, y)
            )
            warmstarts = get_meta_learning_warmstarts(
                task=self.task,
                y=y,
                metafeatures=metafeatures,
                configspace=configspace,
                scoring=self.scoring,
                allowed_models=list(configspace["model_name"].choices),
                top_datasets=self.meta_learning_top_datasets,
                top_configs_per_dataset=self.meta_learning_top_configs_per_dataset,
                max_warmstarts=self.max_meta_learning_warmstarts,
            )
        except Exception:
            return []

        if self.verbose and warmstarts:
            print(f"meta-learning warmstarts loaded: {len(warmstarts)}")
        return warmstarts

    def resolve_per_run_time_limit(self):
        if self.disable_evaluation_timeout:
            return None

        if self.per_run_time_limit is not None:
            return float(self.per_run_time_limit)

        if self.time_budget is None:
            return 60.0

        per_run_time_limit = max(1.0, float(self.time_budget) / 10.0)

        return per_run_time_limit
