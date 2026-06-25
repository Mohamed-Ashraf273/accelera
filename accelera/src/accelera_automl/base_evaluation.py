from abc import ABC
from abc import abstractmethod
from time import perf_counter

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from accelera.src.accelera_automl.configspace_search_space import (
    configuration_space_to_dict,
)


class TrialSpecs:
    def __init__(
        self, stage=0, sample_fraction=1.0, cv_folds=None, model_budget=1.0
    ):
        self.stage = stage
        self.sample_fraction = sample_fraction
        self.cv_folds = cv_folds
        self.model_budget = model_budget


class EvaluationResult(ABC):
    def __init__(
        self,
        model_name=None,
        params=None,
        preprocessing=None,
        score=0.0,
        cost=0.0,
        duration=0.0,
        status=None,
        error=None,
        evaluation_level_stage=0,
        sample_fraction=1.0,
        cv_folds=5,
        model_budget=1.0,
    ):
        self.model_name = model_name
        self.params = params
        self.preprocessing = preprocessing
        self.score = score
        self.cost = cost
        self.duration = duration
        self.status = status
        self.error = error
        self.evaluation_level_stage = evaluation_level_stage
        self.sample_fraction = sample_fraction
        self.cv_folds = cv_folds
        self.model_budget = model_budget


class BaseEvaluation:
    def __init__(
        self,
        *,
        cv=5,
        scoring="accuracy",
        random_state=None,
        n_jobs=None,
    ):
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs

    @abstractmethod
    def make_subsample_splitter(self, target_size):
        "return splitter"

    def get_sample(self, X, y, sample_fraction):
        if sample_fraction == 1.0:
            return X, y

        num_of_samples = len(y)
        target_size = round(num_of_samples * sample_fraction)
        splitter = self.make_subsample_splitter(target_size)
        indices, _ = next(splitter.split(X, y))
        X_sample = X.iloc[indices] if hasattr(X, "iloc") else X[indices]
        y_sample = y.iloc[indices] if hasattr(y, "iloc") else y[indices]
        return X_sample, y_sample

    @abstractmethod
    def return_failure_score(self):
        "return score in case of failture"

    @abstractmethod
    def resolve_cv_folds(self, y, trial_specs):
        "return the effective num of splits for cv"

    @abstractmethod
    def get_cv_splitter(self, cv_num_of_splits):
        "return the splitter"

    @abstractmethod
    def build_estimator(self, model_name, params, trial_specs):
        "return estimator from stored components"

    @abstractmethod
    def get_candidate_preprocessors(model_name, params):
        "return preprocessing cands"

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

    def build_model(
        self, model_name, params, preprocessing="none", trial_specs=None
    ):
        if trial_specs is None:
            trial_specs = TrialSpecs(cv_folds=self.cv)
        estimator = self.build_estimator(model_name, params, trial_specs)
        prepros = self.build_preprocessor(preprocessing)
        return Pipeline(steps=[("preprocessor", prepros), ("estimator", estimator)])

    def evaluate_different_preprocessing(
        self, model_name, params, X, y, trial_specs
    ):
        requested_folds = trial_specs.cv_folds or self.cv
        cv_num_of_splits = self.resolve_cv_folds(y, requested_folds)
        splitter = self.get_cv_splitter(cv_num_of_splits)
        best_preprocessing = "none"
        best_score = float("-inf")

        for preprocessing in self.get_candidate_preprocessors(model_name, params):
            model = self.build_model(
                model_name,
                params,
                preprocessing=preprocessing,
                trial_specs=trial_specs,
            )
            scores = cross_val_score(
                model, X, y, cv=splitter, scoring=self.scoring, n_jobs=self.n_jobs
            )
            score = scores.mean()
            if score > best_score:
                best_score = score
                best_preprocessing = preprocessing

        return best_preprocessing, best_score

    @abstractmethod
    def convert_score_to_cost(self, cost):
        return "convert score to cost so we can work on minimization"

    def evaluate(self, config, X, y, evaluation_level):
        start_time = perf_counter()
        config_dict = configuration_space_to_dict(config)
        model_name = config_dict["model_name"]
        params = config_dict["params"]
        trial_specs = TrialSpecs(
            evaluation_level.stage,
            evaluation_level.sample_fraction,
            evaluation_level.cv_folds,
            evaluation_level.model_budget,
        )
        X_eval, y_eval = self.get_sample(X, y, evaluation_level.sample_fraction)
        try:
            preprocessing_name, score = self.evaluate_different_preprocessing(
                model_name,
                params,
                X_eval,
                y_eval,
                trial_specs=trial_specs,
            )
            status = "success"
            error = None

        except Exception as exc:
            preprocessing_name = "none"
            score = self.return_failure_score()
            status = "failed"
            error = str(exc)

        duration = perf_counter() - start_time
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
            evaluation_level_stage=trial_specs.stage,
            sample_fraction=trial_specs.sample_fraction,
            cv_folds=trial_specs.cv_folds,
            model_budget=trial_specs.model_budget,
        )

    def apply_model_budget(self, estimator, model_name, params, evaluation_level):
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
            minimum, int(round(base_value * evaluation_level.model_budget))
        )

        if hasattr(estimator, "set_params"):
            try:
                estimator.set_params(**{param: scaled_value})
            except ValueError:
                pass
