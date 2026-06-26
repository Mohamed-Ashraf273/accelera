import logging
import sys
from pathlib import Path

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute_array
from amlb.results import save_predictions
from amlb.utils import Timer

log = logging.getLogger(__name__)


def _add_local_automl_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_classifier():
    from accelera_automl.classification import AutoMLClassifier

    return "accelera_automl", AutoMLClassifier


def _load_regressor():
    from accelera_automl.regression import AutoMLRegressor

    return "accelera_automl", AutoMLRegressor


def _map_metric(metric: str) -> str:
    mapping = {
        "acc": "accuracy",
        "auc": "roc_auc",
        "f1": "f1",
        "logloss": "neg_log_loss",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    return mapping.get(metric, "accuracy")


def run(dataset: Dataset, config: TaskConfig):
    _add_local_automl_to_path()

    framework_params = dict(config.framework_params)
    scoring = framework_params.pop("scoring", _map_metric(config.metric))

    if config.type == "classification":
        X_train, X_test = impute_array(dataset.train.X_enc, dataset.test.X_enc)
        y_train, y_test = dataset.train.y_enc, dataset.test.y_enc
        backend_name, Estimator = _load_classifier()
    elif config.type == "regression":
        X_train, X_test = impute_array(dataset.train.X_enc, dataset.test.X_enc)
        y_train = dataset.train.y.squeeze()
        y_test = dataset.test.y.squeeze()
        backend_name, Estimator = _load_regressor()
    else:
        raise ValueError(
            f"MyAutoML adapter currently supports "
            f"classification and regression only, got: {config.type}"
        )

    log.info("\n**** %s [local adapter] ****\n", backend_name)

    common_kwargs = dict(
        time_budget=config.max_runtime_seconds,
        n_trials=framework_params.pop("n_trials", 25),
        cv=framework_params.pop("cv", 3),
        scoring=scoring,
        random_state=config.seed,
        stack_n_jobs=config.cores,
        search_n_parallel=framework_params.pop("search_n_parallel", 1),
        inner_n_jobs=framework_params.pop("inner_n_jobs", 1),
        disable_evaluation_timeout=framework_params.pop(
            "disable_evaluation_timeout", True
        ),
        use_meta_learning=framework_params.pop("use_meta_learning", False),
        verbose=framework_params.pop("verbose", 1),
    )

    predictor = Estimator(
        use_ensemble=framework_params.pop(
            "use_ensemble",
            framework_params.pop("ensemble", True),
        ),
        **common_kwargs,
        **framework_params,
    )

    with Timer() as training:
        predictor.fit(X_train, y_train)
    log.info("Finished fit in %ss.", training.duration)

    with Timer() as predict:
        predictions = predictor.predict(X_test)
    try:
        probabilities = (
            predictor.predict_proba(X_test)
            if config.type == "classification"
            else None
        )
    except AttributeError:
        probabilities = None
    log.info("Finished predict in %ss.", predict.duration)

    save_predictions(
        dataset=dataset,
        output_file=config.output_predictions_file,
        probabilities=probabilities,
        predictions=predictions,
        truth=y_test,
        target_is_encoded=(config.type == "classification"),
    )

    return dict(
        models_count=1,
        duration=training.duration + predict.duration,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )
