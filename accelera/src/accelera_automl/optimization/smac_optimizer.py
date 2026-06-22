from math import ceil
from multiprocessing import Process
from multiprocessing import Queue
from statistics import NormalDist
from time import perf_counter
from time import sleep

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ..evaluation import EvaluationResult
from ..evaluation import TrialSpecs


class OptimizationResult:
    def __init__(self, best_config, best_cost, runhistory, best_config_result):
        self.best_config = best_config
        self.best_cost = best_cost
        self.runhistory = runhistory
        self.best_config_result = best_config_result


class Trial:
    def __init__(self, config, evaluation_level, priority=float("inf")):
        self.config = config
        self.evaluation_level = evaluation_level
        self.priority = priority


class ConfigState:
    def __init__(
        self,
        config,
        signature,
        successful_stages=None,
        stages_this_config_get_promoted_to=None,
    ):
        self.config = config
        self.signature = signature
        self.successful_stages = (
            {} if successful_stages is None else successful_stages
        )
        self.stages_this_config_get_promoted_to = (
            set()
            if stages_this_config_get_promoted_to is None
            else stages_this_config_get_promoted_to
        )


class Optimizer:
    def __init__(
        self,
        *,
        configspace,
        evaluator,
        X,
        y,
        n_trials=50,
        time_budget=None,
        random_state=None,
        per_run_time_limit=None,
        initial_configurations=None,
        n_initial_points=5,
        candidate_pool_size=256,
        verbose=1,
        n_parallel=3,
        evaluation_level=None,
    ):
        self.configspace = configspace
        self.evaluator = evaluator
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.time_budget = time_budget
        self.random_state = random_state
        self.per_run_time_limit = per_run_time_limit
        self.initial_configurations = list(initial_configurations or [])
        self.n_initial_points = max(1, n_initial_points)
        self.candidate_pool_size = max(16, candidate_pool_size)
        self.verbose = verbose
        self.n_parallel = max(1, n_parallel)
        self.evaluation_level = list(
            evaluation_level
            or [
                TrialSpecs(
                    stage=0, sample_fraction=1.0, cv_folds=None, model_budget=1.0
                )
            ]
        )

        self.rng = np.random.default_rng(random_state)
        self.observations = []
        self.best_config = None
        self.best_config_cost = float("inf")
        self.best_config_result = None
        self.seen_signatures = set()
        self.promotion_queue = []
        self.candidate_states = {}
        self.promotion_quantile = 0.35
        self.min_stage_observations_for_promotion = max(3, self.n_parallel + 1)

        if hasattr(self.configspace, "seed"):
            self.configspace.seed(random_state)

    def optimize(self):
        if self.n_trials is None and self.time_budget is None:
            raise ValueError("n_trials=None requires a finite time_budget.")

        started_at = perf_counter()
        trial_id = 0

        while self.n_trials is None or trial_id < self.n_trials:
            if self.time_budget_exceeded(started_at):
                break

            batch_size = (
                self.n_parallel
                if self.n_trials is None
                else min(self.n_parallel, self.n_trials - trial_id)
            )

            trials = self.suggest_batch(batch_size)

            if self.verbose:
                print(
                    f"starting batch of {batch_size} trials "
                    f"({trial_id + 1}/{self.n_trials or 'time budget'})"
                )

            if batch_size > 1:
                batch_results = self.evaluate_batch_with_processes(trials)
            else:
                batch_results = [self.evaluate_single_config(trials[0])]

            for batch_idx, (trial, result) in enumerate(zip(trials, batch_results)):
                current_trial_id = trial_id + batch_idx
                config = trial.config

                if self.verbose:
                    model_name = self.safe_model_name(config)
                    print(
                        f"Trial {current_trial_id + 1}/{self.n_trials} - "
                        f"{model_name}: score={result.score:.6f} "
                        f"cost={result.cost:.6f} status={result.status} "
                        f"stage={result.evaluation_level_stage}"
                    )
                    if result.error and self.verbose > 1:
                        print(
                            f"Trial {current_trial_id + 1}/{self.n_trials} - "
                            f"error: {result.error}"
                        )

                row = {
                    "trial_id": current_trial_id,
                    "config": config,
                    "model_name": result.model_name,
                    "params": result.params,
                    "preprocessing": result.preprocessing,
                    "score": result.score,
                    "cost": result.cost,
                    "duration": result.duration,
                    "status": result.status,
                    "error": result.error,
                    "evaluation_level_stage": result.evaluation_level_stage,
                    "sample_fraction": result.sample_fraction,
                    "cv_folds": result.cv_folds,
                    "model_budget": result.model_budget,
                }
                self.observations.append(row)
                self.record_observation(config, result)

                if (
                    self.is_full_fidelity(result)
                    and result.cost < self.best_config_cost
                ):
                    self.best_config = config
                    self.best_config_cost = float(result.cost)
                    self.best_config_result = result
                    if self.verbose:
                        print(
                            f"New best_config - model={result.model_name} "
                            f"preprocessing={result.preprocessing} "
                            f"score={result.score:.6f}"
                        )

            trial_id += batch_size

        if self.best_config is None and self.observations:
            finite_rows = [
                row for row in self.observations if np.isfinite(row["cost"])
            ]
            if finite_rows:
                best_row = min(finite_rows, key=lambda item: item["cost"])
                self.best_config = best_row["config"]
                self.best_config_cost = float(best_row["cost"])
                self.best_config_result = EvaluationResult(
                    model_name=best_row["model_name"],
                    params=best_row["params"],
                    preprocessing=best_row["preprocessing"],
                    score=best_row["score"],
                    cost=best_row["cost"],
                    duration=best_row["duration"],
                    status=best_row["status"],
                    error=best_row["error"],
                    evaluation_level_stage=best_row.get("evaluation_level_stage", 0),
                    sample_fraction=best_row.get("sample_fraction", 1.0),
                    cv_folds=best_row.get(
                        "cv_folds", self.evaluation_level[-1].cv_folds or 1
                    ),
                    model_budget=best_row.get("model_budget", 1.0),
                )

        return OptimizationResult(
            best_config=self.best_config,
            best_cost=self.best_config_cost,
            runhistory=list(self.observations),
            best_config_result=self.best_config_result,
        )

    def suggest_batch(self, batch_size):
        next_trials = []

        num_of_promoted_trials = self.return_num_of_promoted_trials(batch_size)
        next_trials.extend(self.return_promotions(num_of_promoted_trials))

        # any init configurations first
        for _ in range(batch_size - len(next_trials)):
            initial_config = self.pop_next_initial_configuration()
            if initial_config is not None:
                next_trials.append(
                    Trial(
                        config=initial_config,
                        evaluation_level=self.evaluation_level[0],
                        priority=1.0,
                    )
                )
                self.register_scheduled_trial(next_trials[-1])
            else:
                break

        # if we just started prefer exploration

        if len(self.observations) < self.n_initial_points:
            for _ in range(batch_size - len(next_trials)):
                config = self.sample_random_unseen_configuration()
                next_trials.append(
                    Trial(
                        config=config,
                        evaluation_level=self.evaluation_level[0],
                        priority=1.0,
                    )
                )
                self.register_scheduled_trial(next_trials[-1])
            return next_trials

        # use acquisition function to suggest promising candidates
        finite_observations = self.finite_vectorized_observations()
        if len(finite_observations) < self.n_initial_points:
            for _ in range(batch_size - len(next_trials)):
                config = self.sample_random_unseen_configuration()
                next_trials.append(
                    Trial(
                        config=config,
                        evaluation_level=self.evaluation_level[0],
                        priority=1.0,
                    )
                )
                self.register_scheduled_trial(next_trials[-1])
            return next_trials

        X_obs = np.vstack([row["augmented_vector"] for row in finite_observations])
        y_obs = np.asarray([row["cost"] for row in finite_observations], dtype=float)

        if len(np.unique(y_obs)) <= 1:
            # Not enough variance, sample randomly
            for _ in range(batch_size - len(next_trials)):
                config = self.sample_random_unseen_configuration()
                next_trials.append(
                    Trial(
                        config=config,
                        evaluation_level=self.evaluation_level[0],
                        priority=1.0,  # placeholder
                    )
                )
                self.register_scheduled_trial(next_trials[-1])
            return next_trials

        # fit surrogate model
        surrogate = RandomForestRegressor(
            n_estimators=100,
            bootstrap=True,
            max_features=5 / 6,
            min_samples_split=3,
            min_samples_leaf=3,
            max_depth=20,
            n_jobs=1,
            random_state=self.random_state,
        )
        surrogate.fit(X_obs, y_obs)

        candidate_configs = self.sample_candidate_pool()
        candidate_vectors = np.vstack(
            [
                self.augment_vector(
                    self.config_to_vector(config), self.evaluation_level[0]
                )
                for config in candidate_configs
            ]
        )
        mean, std = self.predict_with_uncertainty(surrogate, candidate_vectors)
        acquisition = self.expected_improvement(
            mean, std, best=self.best_observed_cost()
        )
        ranked_indices = np.argsort(acquisition)[::-1]

        # select top N
        for idx in ranked_indices:
            if len(next_trials) >= batch_size:
                break
            candidate = candidate_configs[int(idx)]
            if self.is_unseen(candidate):
                next_trials.append(
                    Trial(
                        config=candidate,
                        evaluation_level=self.evaluation_level[0],
                        priority=float(mean[int(idx)]),
                    )
                )
                self.register_scheduled_trial(next_trials[-1])

        # If we still need more configurations, sample randomly
        while len(next_trials) < batch_size:
            config = self.sample_random_unseen_configuration()
            next_trials.append(
                Trial(
                    config=config,
                    evaluation_level=self.evaluation_level[0],
                    priority=1.0,  # placeholder
                )
            )
            self.register_scheduled_trial(next_trials[-1])

        return next_trials

    def evaluate_batch_with_processes(self, trials):
        processes = []
        queues = []  # to collect results from processes
        results = [None] * len(trials)
        start_times = {}

        for i, trial in enumerate(trials):
            queue = Queue()
            queues.append(queue)

            process = Process(
                target=self.evaluate_worker,
                args=(queue, trial.config, trial.evaluation_level),
                daemon=True,
            )
            processes.append(process)
            start_times[i] = perf_counter()
            process.start()

        # handle timeout and collect results
        num_of_completed_processes = 0
        remaining_indices = list(range(len(trials)))

        while num_of_completed_processes < len(trials) and remaining_indices:
            current_time = perf_counter()

            # Check for num_of_completed_processes processes
            still_running = []
            for idx in remaining_indices:
                if not queues[idx].empty():
                    try:
                        result = queues[idx].get_nowait()
                        results[idx] = result
                        num_of_completed_processes += 1
                        processes[idx].join()  # clean up
                    except Exception:
                        pass
                else:
                    still_running.append(idx)

            remaining_indices = still_running

            # Kill timed-out processes
            if self.per_run_time_limit is not None:
                for idx in remaining_indices:
                    time_out = (
                        current_time - start_times[idx]
                    )  # Individual process time_out time
                    if time_out >= self.per_run_time_limit:
                        if processes[idx].is_alive():
                            if self.verbose > 1:
                                config_dict = dict(trials[idx].config)
                                model_name = config_dict.get("model_name", "unknown")
                                print(
                                    f"[AutoML] Killing process for {model_name} "
                                    "(timeout)"
                                )
                            processes[idx].terminate()
                            processes[idx].join(timeout=1.0)
                            if processes[idx].is_alive():
                                processes[idx].kill()  # force kill

                            config_dict = dict(trials[idx].config)
                            results[idx] = EvaluationResult(
                                model_name=config_dict.get("model_name", "unknown"),
                                params=config_dict,
                                preprocessing="none",
                                score=0.0,
                                cost=1.0,
                                duration=time_out,
                                status="timeout",
                                error=(
                                    "per_run_time_limit exceeded "
                                    f"({self.per_run_time_limit:.2f}s)"
                                ),
                                evaluation_level_stage=trials[
                                    idx
                                ].evaluation_level.stage,
                                sample_fraction=trials[
                                    idx
                                ].evaluation_level.sample_fraction,
                                cv_folds=trials[idx].evaluation_level.cv_folds or 0,
                                model_budget=trials[
                                    idx
                                ].evaluation_level.model_budget,
                            )
                            num_of_completed_processes += 1

            if remaining_indices:
                sleep(.01)

        return results

    def evaluate_single_config(self, trial):
        if self.per_run_time_limit is None:
            return self.evaluator.direct_evaluation(
                trial.config,
                self.X,
                self.y,
                evaluation_level=trial.evaluation_level,
            )
        return self.evaluate_batch_with_processes([trial])[0]

    def evaluate_worker(self, queue, config, evaluation_level):
        try:
            result = self.evaluate_single_inline(config, evaluation_level)
            queue.put(result)
        except Exception as e:
            config_dict = dict(config)
            queue.put(
                EvaluationResult(
                    model_name=config_dict.get("model_name", "unknown"),
                    params=config_dict,
                    preprocessing="none",
                    score=0.0,
                    cost=1.0,
                    duration=0.0,
                    status="error",
                    error=str(e),
                    evaluation_level_stage=evaluation_level.stage,
                    sample_fraction=evaluation_level.sample_fraction,
                    cv_folds=evaluation_level.cv_folds or 0,
                    model_budget=evaluation_level.model_budget,
                )
            )

    def evaluate_single_inline(self, config, evaluation_level):
        return self.evaluator.evaluate(
            config, self.X, self.y, evaluation_level=evaluation_level
        )

    def vectorized_observations(self):
        rows = []
        for row in self.observations:
            vector = row.get("vector")
            if vector is None:
                vector = self.config_to_vector(row["config"])
                row["vector"] = vector
            augmented_vector = row.get("augmented_vector")
            if augmented_vector is None:
                evaluation_level = TrialSpecs(
                    stage=int(
                        row.get(
                            "evaluation_level_stage", row.get("evaluation_level", 0)
                        )
                    ),
                    sample_fraction=float(row.get("sample_fraction", 1.0)),
                    cv_folds=int(
                        row.get("cv_folds", self.evaluation_level[-1].cv_folds or 1)
                    ),
                    model_budget=float(row.get("model_budget", 1.0)),
                )
                augmented_vector = self.augment_vector(vector, evaluation_level)
                row["augmented_vector"] = augmented_vector
            rows.append(row)
        return rows

    def finite_vectorized_observations(self):
        rows = []
        for row in self.vectorized_observations():
            if not np.isfinite(row.get("cost", np.nan)):
                continue
            augmented_vector = np.asarray(row["augmented_vector"], dtype=float)
            if not np.all(np.isfinite(augmented_vector)):
                continue
            rows.append(row)
        return rows

    def sample_random_configuration(self):
        return self.configspace.sample_configuration()

    def sample_random_unseen_configuration(self):
        max_attempts = max(self.candidate_pool_size * 8, 128)
        for _ in range(max_attempts):
            config = self.sample_random_configuration()
            if self.is_unseen(config):
                return config
        return self.sample_random_configuration()

    def sample_candidate_pool(self):
        candidates = []
        seen = set()
        max_attempts = self.candidate_pool_size * 4

        for _ in range(max_attempts):
            config = self.sample_random_configuration()
            vector_key = tuple(self.config_to_vector(config).tolist())
            if vector_key in seen:
                continue
            seen.add(vector_key)
            candidates.append(config)
            if len(candidates) >= self.candidate_pool_size:
                break

        if not candidates:
            candidates.append(self.sample_random_configuration())

        return candidates

    def best_observed_cost(self):
        finite_costs = [
            row["cost"]
            for row in self.observations
            if np.isfinite(row.get("cost", np.nan))
        ]
        if not finite_costs:
            return float("inf")
        return float(min(finite_costs))

    @staticmethod
    def predict_with_uncertainty(
        surrogate,
        X,
    ):
        tree_predictions = np.vstack(
            [tree.predict(X) for tree in surrogate.estimators_]
        )
        mean = tree_predictions.mean(axis=0)
        std = tree_predictions.std(axis=0)
        std = np.maximum(std, 1e-9)
        return mean, std

    @staticmethod
    def expected_improvement(
        mean,
        std,
        *,
        best,
        xi=0.01,
    ):
        improvement = best - mean - xi
        z = improvement / std
        normal = NormalDist()
        phi = np.asarray([normal.pdf(float(value)) for value in z], dtype=float)
        Phi = np.asarray([normal.cdf(float(value)) for value in z], dtype=float)
        ei = improvement * Phi + std * phi
        ei[std <= 1e-12] = 0.0
        return ei

    @staticmethod
    def config_to_vector(config):
        if hasattr(config, "get_array"):
            vector = np.asarray(config.get_array(), dtype=float)
        else:
            raise TypeError("Configuration object must implement `get_array()`.")

        return np.nan_to_num(vector, nan=-1.0, posinf=1.0, neginf=-1.0)

    def config_signature(self, config):
        return tuple(self.config_to_vector(config).tolist())

    @staticmethod
    def augment_vector(config_vector, evaluation_level):
        cv_component = (
            0.0
            if evaluation_level.cv_folds is None
            else float(evaluation_level.cv_folds)
        )
        evaluation_level_vector = np.asarray(
            [
                float(evaluation_level.stage),
                float(evaluation_level.sample_fraction),
                cv_component,
                float(evaluation_level.model_budget),
            ],
            dtype=float,
        )
        return np.concatenate([config_vector, evaluation_level_vector], dtype=float)

    def is_unseen(self, config):
        return self.config_signature(config) not in self.seen_signatures

    def mark_seen(self, config):
        self.seen_signatures.add(self.config_signature(config))

    def record_observation(
        self,
        config,
        result,
    ):
        signature = self.config_signature(config)
        state = self.candidate_states.get(signature)
        if state is None:
            state = ConfigState(config=config, signature=signature)
            self.candidate_states[signature] = state

        current_stage = int(result.evaluation_level_stage)

        if result.status == "success":
            state.successful_stages[current_stage] = float(result.cost)

        next_stage = current_stage + 1
        if next_stage >= len(self.evaluation_level):
            return
        if result.status != "success":
            return
        if next_stage in state.stages_this_config_get_promoted_to:
            return
        if not self.should_promote(state, current_stage):
            return

        promoted_trial = Trial(
            config=config,
            evaluation_level=self.evaluation_level[next_stage],
            priority=float(result.cost),
        )

        self.promotion_queue.append(promoted_trial)
        self.promotion_queue.sort(
            key=lambda trial: (trial.evaluation_level.stage, trial.priority)
        )
        state.stages_this_config_get_promoted_to.add(next_stage)

    def should_promote(self, candidate_state, stage):
        if stage not in candidate_state.successful_stages:
            return False
        eligible_states = [
            state
            for state in self.candidate_states.values()
            if stage in state.successful_stages
        ]
        if not eligible_states:
            return True
        if len(eligible_states) < self.min_stage_observations_for_promotion:
            return True

        ranked_states = sorted(
            eligible_states,
            key=lambda state: state.successful_stages[stage],
        )
        promotion_count = max(1, ceil(len(ranked_states) * self.promotion_quantile))
        promoted_signatures = {
            state.signature for state in ranked_states[:promotion_count]
        }
        return candidate_state.signature in promoted_signatures

    def is_full_fidelity(self, result):
        stage = getattr(
            result, "evaluation_level_stage", getattr(result, "fidelity_stage", 0)
        )
        return int(stage) >= (len(self.evaluation_level) - 1)

    def register_scheduled_trial(self, trial):
        signature = self.config_signature(trial.config)
        if trial.evaluation_level.stage == 0:
            self.mark_seen(trial.config)

        state = self.candidate_states.get(signature)
        if state is None:
            state = ConfigState(config=trial.config, signature=signature)
            self.candidate_states[signature] = state

    def return_promotions(self, limit):
        if limit <= 0 or not self.promotion_queue:
            return []

        taken = self.promotion_queue[:limit]
        self.promotion_queue = self.promotion_queue[limit:]

        for trial in taken:
            self.register_scheduled_trial(trial)

        return taken

    def return_num_of_promoted_trials(self, batch_size):
        if not self.promotion_queue:
            return 0

        if (
            len(self.observations) < self.n_initial_points
        ):  # don't promote during initial exploration phase
            return 0

        num = len(self.promotion_queue)
        if self.best_config_result is None:
            return min(num, max(1, batch_size // 2))
        return min(num, max(1, (2 * batch_size) // 3))

    def pop_next_initial_configuration(self):
        while self.initial_configurations:
            config = self.initial_configurations.pop(0)
            if self.is_unseen(config):
                return config
        return None

    def time_budget_exceeded(self, started_at):
        if self.time_budget is None:
            return False
        return (perf_counter() - started_at) >= float(self.time_budget)

    @staticmethod
    def safe_model_name(config):
        try:
            config_dict = dict(config)
            return str(config_dict.get("model_name", "<unknown>"))
        except Exception:
            return "<unknown>"
