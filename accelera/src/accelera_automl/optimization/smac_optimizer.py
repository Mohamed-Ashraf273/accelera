from math import ceil
from multiprocessing import Process
from multiprocessing import Queue
from statistics import NormalDist
from time import perf_counter
from time import sleep

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from accelera.src.accelera_automl.base_evaluation import EvaluationResult
from accelera.src.accelera_automl.base_evaluation import TrialSpecs


class OptimizationResult:
    def __init__(self, best_config, best_cost, trails_history, best_config_trial):
        self.best_config = best_config
        self.best_cost = best_cost
        self.trails_history = trails_history
        self.runhistory = trails_history
        self.best_config_trial = best_config_trial
        self.best_config_result = best_config_trial


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
        self.successful_stages = successful_stages or {}
        self.stages_this_config_get_promoted_to = (
            stages_this_config_get_promoted_to or set()
        )


class Optimizer:
    def __init__(
        self,
        *,
        config_space=None,
        configspace=None,
        evaluator,
        X,
        y,
        time_budget,
        per_trial_timelimit=None,
        evaluation_level=None,
        n_trials=None,
        random=None,
        warm_start_configs=[],
        num_of_initail_points_to_try=5,
        sample_size_from_configspace=256,
        num_of_trials_to_run_parralel=3,
        n_parallel=None,
        verbose=1,
    ):
        self.config_space = config_space if config_space is not None else configspace
        self.evaluator = evaluator
        self.X = X
        self.y = y
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.random = random
        self.per_trial_timelimit = per_trial_timelimit
        self.warm_start_configs = warm_start_configs
        self.num_of_initail_points_to_try = num_of_initail_points_to_try
        self.sample_size_from_configspace = sample_size_from_configspace
        self.verbose = verbose
        self.num_of_trials_to_run_parralel = (
            n_parallel if n_parallel is not None else num_of_trials_to_run_parralel
        )
        self.evaluation_level = evaluation_level or [
            TrialSpecs(stage=0, sample_fraction=1.0, cv_folds=5, model_budget=1.0)
        ]

        # optimization output
        self.trials = []
        self.best_config = None
        self.best_cost = float("inf")
        self.best_config_trial = None

        self.seen_signatures = set()
        self.promotion_queue = []
        self.candidate_states = {}
        self.promotion_quantile = 0.4
        self.min_stage_observations_for_promotion = max(
            3, self.num_of_trials_to_run_parralel + 1
        )

        if hasattr(self.config_space, "seed"):
            self.config_space.seed(random)

    def optimize(self):
        if self.n_trials is None and self.time_budget is None:
            raise ValueError("n_trials=None requires a finite time_budget.")

        started_at = perf_counter()
        trial_num = 0

        while self.n_trials is None or trial_num < self.n_trials:
            if self.time_budget_exceeded(started_at):
                break

            batch_size = self.num_of_trials_to_run_parralel
            if self.n_trials is not None:
                batch_size = min(batch_size, self.n_trials - trial_num)
            trials = self.suggest_batch(batch_size)
            if self.verbose:
                print(
                    f"Starting optimization batch with {len(trials)} trial(s) "
                    f"at trial {trial_num}."
                )

            if self.num_of_trials_to_run_parralel > 1:
                batch_results = self.evaluate_batch(trials)
            else:
                batch_results = [self.evaluate_single_config(trials[0])]

            for idx, (trial, result) in enumerate(zip(trials, batch_results)):
                current_trial_num = trial_num + idx
                trial_dict = {
                    "trial_id": current_trial_num,
                    "config": trial.config,
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

                self.trials.append(trial_dict)
                self.record_trial(trial.config, result)
                if self.verbose:
                    print(
                        f"Trial {current_trial_num}: "
                        f"{result.model_name} status={result.status} "
                        f"score={result.score}"
                    )
                    if result.error:
                        print(f"Trial {current_trial_num} error: {result.error}")

                if self.is_full_fidelity(result) and result.cost < self.best_cost:
                    self.best_config = trial.config
                    self.best_cost = result.cost
                    self.best_config_trial = result
                    if self.verbose:
                        print(
                            f"New best configuration at trial {current_trial_num}: "
                            f"cost={self.best_cost}"
                        )
            trial_num += self.num_of_trials_to_run_parralel

        if self.best_config is None:
            successful_trials = [
                trial for trial in self.trials if trial.get("status") == "success"
            ]
            if successful_trials:
                best_trial = min(successful_trials, key=lambda trial: trial["cost"])
                self.best_config = best_trial["config"]
                self.best_cost = best_trial["cost"]
                self.best_config_trial = EvaluationResult(
                    model_name=best_trial["model_name"],
                    params=best_trial["params"],
                    preprocessing=best_trial["preprocessing"],
                    score=best_trial["score"],
                    cost=best_trial["cost"],
                    duration=best_trial["duration"],
                    status=best_trial["status"],
                    error=best_trial["error"],
                    evaluation_level_stage=best_trial["evaluation_level_stage"],
                    sample_fraction=best_trial["sample_fraction"],
                    cv_folds=best_trial["cv_folds"],
                    model_budget=best_trial["model_budget"],
                )

        return OptimizationResult(
            self.best_config,
            self.best_cost,
            self.trials,
            self.best_config_trial,
        )

    def is_full_fidelity(self, result):
        stage = result.evaluation_level_stage
        return stage == (len(self.evaluation_level) - 1)

    def should_promote(self, state, cur_stage):
        states = [
            state
            for state in self.candidate_states.values()
            if cur_stage in state.successful_stages
        ]

        if not states:
            return True

        if len(states) < self.min_stage_observations_for_promotion:
            return True

        ranked_states = sorted(
            states, key=lambda state: state.successful_stages[cur_stage]
        )
        promotion_count = max(1, ceil(len(ranked_states) * self.promotion_quantile))
        promoted_signs = {
            state.signature for state in ranked_states[:promotion_count]
        }

        return state.signature in promoted_signs

    def record_trial(self, config, result):
        sign = self.config_to_sign(config)
        state = self.candidate_states.get(sign)
        if state is None:
            state = ConfigState(config, sign)
            self.candidate_states[sign] = state

        cur_stage = result.evaluation_level_stage

        if result.status == "success":
            state.successful_stages[cur_stage] = result.cost

        next_stage = cur_stage + 1
        if next_stage >= len(self.evaluation_level):
            return

        if result.status != "success":
            return

        if next_stage in state.stages_this_config_get_promoted_to:
            return

        if not self.should_promote(state, cur_stage):
            return

        promoted_trial = Trial(
            config, self.evaluation_level[next_stage], result.cost
        )
        self.promotion_queue.append(promoted_trial)
        self.promotion_queue.sort(
            key=lambda trial: (trial.evaluation_level.stage, trial.priority)
        )
        state.stages_this_config_get_promoted_to.add(next_stage)

    def evaluate_single(self, trial):
        if self.per_trial_timelimit is None:
            return self.evaluator.evaluate(
                trial.config,
                self.X,
                self.y,
                evaluation_level=trial.evaluation_level,
            )
        return self.evaluate_batch([trial])[0]

    def evaluate_single_config(self, trial):
        return self.evaluate_single(trial)

    def evaluate_worker(self, queue, config, evaluation_level):
        try:
            result = self.evaluate_single_inline(config, evaluation_level)
            queue.put(result)
        except Exception as e:
            config_dict = dict(config)
            queue.put(
                EvaluationResult(
                    model_name=config_dict.get("model_name"),
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

    def evaluate_batch(self, trials):
        if self.per_trial_timelimit is None:
            return [
                self.evaluate_single_inline(trial.config, trial.evaluation_level)
                for trial in trials
            ]

        processes = []
        queues = []
        results = [None] * len(trials)
        start_times = {}

        for i, trial in enumerate(trials):
            queue = Queue()
            queues.append(queue)

            process = Process(
                target=self.evaluate_worker,
                args=(queue, trial.config, trial.evaluation_level),
            )
            processes.append(process)
            start_times[i] = perf_counter()
            process.start()

        num_of_completed_processes = 0
        remaining = list(range(len(trials)))

        while num_of_completed_processes < len(trials):
            current_time = perf_counter()

            still_running = []
            for idx in remaining:
                if not queues[idx].empty():
                    result = queues[idx].get_nowait()
                    results[idx] = result
                    num_of_completed_processes += 1
                    processes[idx].join()  # clean up

                else:
                    still_running.append(idx)

            remaining = still_running

            if self.per_trial_timelimit is not None:
                for idx in remaining:
                    timeout = current_time - start_times[idx]
                    if timeout >= self.per_trial_timelimit:
                        config_dict = dict(trials[idx].config)
                        processes[idx].terminate()
                        processes[idx].join(timeout=1.0)
                        if processes[idx].is_alive():
                            processes[idx].kill()  # force kill
                        if self.verbose:
                            print(
                                f"Killed timed out trial {idx} after {timeout:.2f}s."
                            )

                        results[idx] = EvaluationResult(
                            model_name=config_dict.get("model_name"),
                            params=config_dict,
                            preprocessing="none",
                            score=0.0,
                            cost=1.0,
                            duration=timeout,
                            status="timeout",
                            error=(
                                "per_run_time_limit exceeded "
                                f"({self.per_trial_timelimit:.2f}s)"
                            ),
                            evaluation_level_stage=trials[
                                idx
                            ].evaluation_level.stage,
                            sample_fraction=trials[
                                idx
                            ].evaluation_level.sample_fraction,
                            cv_folds=trials[idx].evaluation_level.cv_folds or 0,
                            model_budget=trials[idx].evaluation_level.model_budget,
                        )
                        num_of_completed_processes += 1

            if remaining:
                sleep(0.01)

        return results

    def time_budget_exceeded(self, start_time):
        if self.time_budget is None:
            return False

        return (perf_counter() - start_time) > self.time_budget

    def get_num_of_promoted_trials(self):
        if len(self.promotion_queue) == 0:
            return 0

        if len(self.trials) < self.num_of_initail_points_to_try:
            return 0  # don't promote during initial exploration phase

        if (
            self.best_config_trial is None
        ):  # still exploration, small promotion percentage
            return min(
                len(self.promotion_queue),
                max(1, self.num_of_trials_to_run_parralel // 2),
            )

        return min(
            len(self.promotion_queue),
            max(1, (2 * self.num_of_trials_to_run_parralel) // 3),
        )  # go into exploitation

    def config_to_vector(self, config):
        vector = np.asarray(config.get_array(), dtype=float)
        return np.nan_to_num(vector, nan=-1.0, posinf=1.0, neginf=-1.0)

    def config_to_sign(self, config):
        return tuple(self.config_to_vector(config).tolist())

    def mark_seen(self, sign):
        self.seen_signatures.add(sign)

    def register_trial(self, trial):
        sign = self.config_to_sign(trial.config)
        if trial.evaluation_level.stage == 0:
            self.mark_seen(sign)

        state = self.candidate_states.get(sign)
        if state is None:
            state = ConfigState(trial.config, sign)
            self.candidate_states[sign] = state

    def get_promoted_trials(self, num_of_promoted_trials):
        if num_of_promoted_trials == 0:
            return []

        promoted_trails = self.promotion_queue[:num_of_promoted_trials]
        self.promotion_queue = self.promotion_queue[num_of_promoted_trials:]

        for trial in promoted_trails:
            self.register_trial(trial)

        return promoted_trails

    def is_unseen(self, config):
        return self.config_to_sign(config) not in self.seen_signatures

    def pop_next_warmstart_config(self):
        while self.warm_start_configs:
            config = self.warm_start_configs.pop(0)
            if self.is_unseen(config):
                return config

            return None

    def sample_random_configuration(self):
        return self.config_space.sample_configuration()

    def sample_random_unseen_configuration(self):
        max_attempts = self.sample_size_from_configspace * 8
        for _ in range(max_attempts):
            config = self.sample_random_configuration()
            if self.is_unseen(config):
                return config

        return self.sample_random_configuration()

    def suggest_batch(self, batch_size=None):
        next_trials = []
        num = batch_size or self.num_of_trials_to_run_parralel
        num_of_promoted_trials = self.get_num_of_promoted_trials()
        next_trials.extend(self.get_promoted_trials(num_of_promoted_trials))

        # try warmstart config first
        for _ in range(num - len(next_trials)):
            config = self.pop_next_warmstart_config()
            if config is not None:
                next_trials.append(Trial(config, self.evaluation_level[0]))
                self.register_trial(next_trials[-1])
            else:
                break

        # prefer exploration until we reach num_of_initail_points_to_try
        if len(self.trials) < self.num_of_initail_points_to_try:
            for _ in range(num - len(next_trials)):
                config = self.sample_random_unseen_configuration()
                next_trials.append(Trial(config, self.evaluation_level[0]))
                self.register_trial(next_trials[-1])

            return next_trials

        finite_trials = self.get_finite_vectors()

        if len(finite_trials) < self.num_of_initail_points_to_try:
            for _ in range(num - len(next_trials)):
                config = self.sample_random_unseen_configuration()
                next_trials.append(Trial(config, self.evaluation_level[0]))
                self.register_trial(next_trials[-1])

            return next_trials

        X_obs = np.vstack([trial["augmented_vector"] for trial in finite_trials])
        y = np.asarray([trial["cost"] for trial in finite_trials], dtype=float)

        # fit surrogate model
        surrogate = RandomForestRegressor(
            n_estimators=100,
            bootstrap=True,
            max_features=1.0,
            min_samples_split=3,
            min_samples_leaf=3,
            max_depth=20,
            n_jobs=1,
            random_state=self.random,
        )

        surrogate.fit(X_obs, y)
        sample_configs = self.sample_configurations()
        sample_configs_vectors = np.vstack(
            [
                self.augment_vector(
                    self.config_to_vector(config), self.evaluation_level[0]
                )
                for config in sample_configs
            ]
        )

        mean, std = self.predict_using_surregate(surrogate, sample_configs_vectors)
        acquisition = self.expected_improvement(mean, std, self.get_best_cost())
        ranked_indices = np.argsort(acquisition)[::-1]

        for idx in ranked_indices:
            if len(next_trials) >= num:
                break

            candidate_config = sample_configs[idx]
            if self.is_unseen(candidate_config):
                next_trials.append(Trial(candidate_config, self.evaluation_level[0]))
            self.register_trial(next_trials[-1])

        while len(next_trials) < num:
            config = self.sample_random_unseen_configuration()
            next_trials.append(Trial(config, self.evaluation_level[0]))
            self.register_trial(next_trials[-1])

        return next_trials

    def expected_improvement(self, mean, std, best_cost):
        xi = 0.01
        improvement = best_cost - mean - xi
        normalized_improvement = improvement / std
        normal = NormalDist()
        phi = np.asarray(
            [normal.pdf(float(value)) for value in normalized_improvement],
            dtype=float,
        )
        Phi = np.asarray(
            [normal.cdf(float(value)) for value in normalized_improvement],
            dtype=float,
        )
        ei = improvement * Phi + std * phi
        ei[std <= 1e-12] = 0.0
        return ei

    def predict_using_surregate(self, surrogate, X):
        trees_prediction = np.vstack(
            [tree.predict(X) for tree in surrogate.estimators_]
        )
        mean = trees_prediction.mean(axis=0)
        std = trees_prediction.std(axis=0)
        std = np.maximum(std, 1e-9)
        return mean, std

    def get_best_cost(self):
        finite_costs = [
            row["cost"]
            for row in self.trials
            if np.isfinite(row.get("cost", np.nan))
        ]
        if not finite_costs:
            return float("inf")
        return float(min(finite_costs))

    def sample_configurations(self):
        candidates = []
        seen = set()
        max_attempts = self.sample_size_from_configspace * 4

        for _ in range(max_attempts):
            config = self.sample_random_configuration()
            sign = self.config_to_sign(config)
            if sign in seen or not self.is_unseen(config):
                continue

            seen.add(sign)
            candidates.append(config)

            if len(candidates) == self.sample_size_from_configspace:
                break

        if not candidates:
            candidates.append(self.sample_random_configuration())

        return candidates

    def augment_vector(self, config_vector, trial_specs):
        vector = np.asarray(
            [
                trial_specs.stage,
                trial_specs.sample_fraction,
                0.0 if trial_specs.cv_folds is None else trial_specs.cv_folds,
                trial_specs.model_budget,
            ],
            dtype=float,
        )

        return np.concatenate([config_vector, vector], dtype=float)

    def vectorize_observations(self):
        trials = []
        for trial in self.trials:
            vector = trial.get("vector")
            if vector is None:
                vector = self.config_to_vector(trial["config"])
                trial["vector"] = vector

            augmented_vector = trial.get("augmented_vector")
            if augmented_vector is None:
                trial_specs = TrialSpecs(
                    trial.get("evaluation_level_stage"),
                    trial.get("sample_fraction"),
                    trial.get("cv_folds"),
                    trial.get("model_budget"),
                )
                augmented_vector = self.augment_vector(vector, trial_specs)
                trial["augmented_vector"] = augmented_vector

            trials.append(trial)
        return trials

    def get_finite_vectors(self):
        trials = []

        for trial in self.vectorize_observations():
            if not np.isfinite(trial.get("cost", np.nan)):
                continue

            augmented_vector = np.asanyarray(trial["augmented_vector"], dtype=float)
            if not np.all(np.isfinite(augmented_vector)):
                continue

            trials.append(trial)

        return trials
