import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor

from accelera.src.accelera_automl.base_evaluation import TrialSpecs
from accelera.src.accelera_automl.configspace_search_space import (
    configuration_space_to_dict,
)
from accelera.src.accelera_automl.configspace_search_space import (
    return_classification_config_space,
)
from accelera.src.accelera_automl.configspace_search_space import (
    return_regression_config_space,
)
from accelera.src.accelera_automl.evaluation import ClassificationEvaluator
from accelera.src.accelera_automl.evaluation import RegressionEvaluator
from accelera.src.accelera_automl.meta_learning import (
    compute_basic_classification_metafeatures,
)
from accelera.src.accelera_automl.meta_learning import (
    compute_basic_regression_metafeatures,
)
from accelera.src.accelera_automl.meta_learning import get_meta_learning_warmstarts
from accelera.src.accelera_automl.optimization.smac_optimizer import Optimizer
from accelera.src.accelera_automl.stacked_ensemble import StackedEnsembleClassifier
from accelera.src.accelera_automl.stacked_ensemble_regression import (
    StackedEnsembleRegressor,
)


class AutoMLResult:
    def __init__(
        self,
        leaderboard=None,
        final_model=None,
        best_model=None,
        best_score=None,
        ensemble=None,
        runhistory=None,
    ):
        self.best_model = best_model
        self.best_score = best_score
        self.ensemble = ensemble
        self.runhistory = runhistory
        self.leaderboard = leaderboard
        self.final_model = final_model


Automlresult = AutoMLResult


fidality_stages = (
    TrialSpecs(stage=0, sample_fraction=0.2, cv_folds=2, model_budget=0.25),
    TrialSpecs(stage=1, sample_fraction=0.5, cv_folds=3, model_budget=0.6),
    TrialSpecs(stage=2, sample_fraction=1.0, cv_folds=None, model_budget=1.0),
)


class AutoMLEngine:
    def __init__(
        self,
        *,
        time_budget=None,
        task,
        per_run_time_limit=None,
        n_trials=50,
        scoring=None,
        cv=5,
        use_ensemble=True,
        random_state=None,
        ensemble_strategy="stacked",
        stacked_base_size=4,
        ensemble_voting_size=10,
        stacked_include_original_features_in_meta=False,
        search_n_parallel=3,
        stacked_bagging_n_estimators=5,
        inner_n_jobs=1,
        stack_n_jobs=None,
        balance_classes=False,
        use_meta_learning=True,
        allowed_models=None,
        meta_learning_top_datasets=5,
        disable_evaluation_timeout=False,
        meta_learning_top_configs_per_dataset=3,
        candidate_pool_size=256,
        max_meta_learning_warmstarts=10,
        n_initial_points=5,
        verbose=1,
    ):
        self.task = task
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.scoring = scoring
        self.ensemble_voting_size = ensemble_voting_size
        self.ensemble = use_ensemble
        self.use_meta_learning = use_meta_learning
        self.per_run_time_limit = per_run_time_limit
        self.ensemble_strategy = ensemble_strategy
        self.stacked_base_size = stacked_base_size
        self.stacked_bagging_n_estimators = stacked_bagging_n_estimators
        self.stacked_include_original_features_in_meta = (
            stacked_include_original_features_in_meta
        )
        self.search_n_parallel = search_n_parallel
        self.stack_n_jobs = (
            1 if stack_n_jobs is None or stack_n_jobs < 1 else stack_n_jobs
        )
        self.inner_n_jobs = inner_n_jobs
        self.disable_evaluation_timeout = disable_evaluation_timeout
        self.allowed_models = allowed_models
        self.balance_classes = balance_classes
        self.meta_learning_top_configs_per_dataset = (
            meta_learning_top_configs_per_dataset
        )
        self.n_initial_points = n_initial_points
        self.meta_learning_top_datasets = meta_learning_top_datasets
        self.candidate_pool_size = candidate_pool_size
        self.max_meta_learning_warmstarts = max_meta_learning_warmstarts
        self.verbose = verbose

    def build_evaluator(self):
        if self.task == "classification":
            return ClassificationEvaluator(
                cv=self.cv,
                scoring=self.scoring,
                random_state=self.random_state,
                n_jobs=self.inner_n_jobs,
                balance_classes=self.balance_classes,
            )
        return RegressionEvaluator(
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=self.inner_n_jobs,
        )

    def resolve_per_run_time_limit(self):
        if self.disable_evaluation_timeout:
            return None

        if self.per_run_time_limit is not None:
            return float(self.per_run_time_limit)

        if self.time_budget is None:
            return 60.0

        per_run_time_limit = max(1.0, float(self.time_budget) / 10.0)

        return per_run_time_limit

    def get_optimizer(
        self,
        configspace,
        evaluator,
        X,
        y,
        warm_start,
    ):
        return Optimizer(
            config_space=configspace,
            evaluator=evaluator,
            X=X,
            y=y,
            n_trials=self.n_trials,
            time_budget=self.time_budget,
            random=self.random_state,
            per_trial_timelimit=self.resolve_per_run_time_limit(),
            warm_start_configs=warm_start,
            num_of_trials_to_run_parralel=self.search_n_parallel,
            evaluation_level=list(fidality_stages),
            sample_size_from_configspace=self.candidate_pool_size,
            num_of_initail_points_to_try=self.n_initial_points,
            verbose=self.verbose,
        )

    def build_leaderboard(self, result):
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
                    "trial_id": row.get("trial_id"),
                    "model_name": row["model_name"],
                    "preprocessing": row.get("preprocessing", "none"),
                    "cv_score": row["score"],
                    "error": row.get("error"),
                    "params": row["params"],
                }
            )

        return leaderboard

    def fit_final_model(
        self,
        best_config,
        evaluator,
        X,
        y,
        preprocessing="none",
    ):
        config_dict = configuration_space_to_dict(best_config)
        model = evaluator.build_model(
            config_dict["model_name"],
            config_dict["params"],
            preprocessing=preprocessing,
        )
        model.fit(X, y)
        return model

    def resolve_ensemble_strategy(self, X):
        n_rows = len(X)
        if self.ensemble_strategy == "stacked" and n_rows > 50_000:
            if self.verbose:
                print(
                    "Large dataset detected; using "
                    "voting ensemble instead of stacking."
                )
            return "voting"
        return self.ensemble_strategy

    def get_voting_strategy(self, estimators):
        if all(hasattr(estimator[1], "predict_proba") for estimator in estimators):
            return "soft"
        return "hard"

    def make_voting_classifier(self, estimators):
        return VotingClassifier(
            estimators=estimators,
            voting=self.get_voting_strategy(estimators),
            n_jobs=self.inner_n_jobs,
        )

    def build_ensemble(self, results, evaluator, X, y):
        if not self.ensemble:
            return None, None

        strategy = self.resolve_ensemble_strategy(X)
        if strategy == "stacked":
            return self.build_stacked_ensemble(results, evaluator, X, y)
        if strategy != "voting":
            raise ValueError("unsupported ensemble strategy.")
        if self.task == "regression":
            return self.build_regression_voting_ensemble(results, evaluator, X, y)

        ranked = self.rank_successful(results)
        if len(ranked) < 2:
            return None, None

        candidates = self.build_candidate_pool(ranked, evaluator)
        if len(candidates) < 2:
            return None, None

        selected, score = self.greedy_forward_select(
            candidates, self.make_voting_classifier, X, y
        )
        if len(selected) < 2:
            return None, None

        model = self.make_voting_classifier(selected)
        model.fit(X, y)
        return model, float(score)

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
                    "stacked_bagging_n_estimators": (
                        self.stacked_bagging_n_estimators
                    ),
                    "include_original_features_in_meta": (
                        self.stacked_include_original_features_in_meta
                    ),
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

        if self.verbose:
            print(
                "AutoML disabled models for dataset size/features: "
                f"{sorted(set(disabled_models))}"
            )

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
                model_name
                for model_name in candidate_models
                if model_name not in disabled_set
            ]

        return filtered_models

    def features_have_negative_values(self, X):
        try:
            return bool((X < 0).any().any())
        except AttributeError:
            return bool(np.asarray(X).min() < 0)

    def build_configspace(self, models):
        if self.task == "classification":
            return return_classification_config_space(allowed_models=models)
        return return_regression_config_space(allowed_models=models)

    def get_warmstart(self, configspace, X, y):
        if not self.use_meta_learning:
            if self.verbose:
                print("Meta-learning warmstart disabled.")
            return []

        try:
            if self.verbose:
                print("Starting meta-learning warmstart selection.")
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
        except Exception as exc:
            if self.verbose:
                print(f"Meta-learning warmstart skipped: {exc}")
            return []

        if self.verbose:
            print(
                "Meta-learning warmstart selected "
                f"{len(warmstarts)} configuration(s)."
            )
        return warmstarts

    def search(self, X, y):
        allowed_models = self.resolve_allowed_models(X)
        config_space = self.build_configspace(allowed_models)
        warm_start = self.get_warmstart(config_space, X, y)
        evaluator = self.build_evaluator()
        optimizer = self.get_optimizer(config_space, evaluator, X, y, warm_start)
        optimization_result = optimizer.optimize()
        best_config = optimization_result.best_config
        if best_config is None:
            raise RuntimeError("optimizer did not return a best configuration.")
        best_config_result = optimization_result.best_config_result
        preprocessing = best_config_result.preprocessing
        leaderboard = self.build_leaderboard(optimization_result.runhistory)

        best_model = self.fit_final_model(
            best_config, evaluator, X, y, preprocessing=preprocessing
        )
        ensemble, ensemble_score = self.build_ensemble(
            optimization_result.runhistory,
            evaluator,
            X,
            y,
        )
        best_model_score = float(best_config_result.score)
        final_model = best_model
        final_score = best_model_score

        if ensemble is not None and ensemble_score is not None:
            if ensemble_score > best_model_score:
                final_model = ensemble
                final_score = ensemble_score

            leaderboard = self.append_ensemble_to_leaderboard(
                leaderboard=leaderboard,
                ensemble_score=ensemble_score,
            )

        return AutoMLResult(
            leaderboard=leaderboard,
            final_model=final_model,
            best_model=best_model,
            best_score=final_score,
            ensemble=ensemble,
            runhistory=optimization_result.runhistory,
        )

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

        if self.task == "regression":
            ensemble_model = StackedEnsembleRegressor(
                base_estimators=base_estimators,
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

    def make_voting_regressor(self, estimators):
        return VotingRegressor(estimators=estimators)

    def build_candidate_pool(self, ranked_rows, evaluator):
        candidates = []
        seen = set()
        cap = self.ensemble_voting_size * 3

        for row in ranked_rows:
            cfg = configuration_space_to_dict(row["config"])
            sig = (
                cfg["model_name"],
                tuple(sorted(cfg["params"].items())),
                row.get("preprocessing", "none"),
            )
            if sig in seen:
                continue

            estimator = evaluator.build_model(
                cfg["model_name"],
                cfg["params"],
                preprocessing=row.get("preprocessing", "none"),
            )
            candidates.append((f"{cfg['model_name']}_{len(candidates)}", estimator))
            seen.add(sig)

            if len(candidates) == cap:
                break

        return candidates

    def build_regression_voting_ensemble(self, results, evaluator, X, y):
        ranked = self.rank_successful(results)
        if len(ranked) < 2:
            return None, None

        candidates = self.build_candidate_pool(ranked, evaluator)
        if len(candidates) < 2:
            return None, None

        selected, score = self.greedy_forward_select(
            candidates, self.make_voting_regressor, X, y
        )
        if len(selected) < 2:
            return None, None

        model = self.make_voting_regressor(selected)
        model.fit(X, y)
        return model, float(score)

    def rank_successful(self, results):
        return [
            result
            for result in sorted(
                results,
                key=lambda r: r.get("score", float("-inf")),
                reverse=True,
            )
            if result.get("status") == "success" and result.get("config") is not None
        ]

    def greedy_forward_select(self, candidates, make_ensemble, X, y):
        selected = []
        best_score = float("-inf")
        pool = list(candidates)

        while pool and len(selected) < self.ensemble_voting_size:
            best_idx = None
            best_candidate_score = best_score

            for idx, (name, est) in enumerate(pool):
                trial = make_ensemble(selected + [(name, est)])
                score = self.evaluate_estimator(trial, X, y)
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_idx = idx

            if best_idx is None:
                break

            selected.append(pool.pop(best_idx))
            best_score = best_candidate_score

        return selected, best_score
