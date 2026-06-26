from accelera.src.accelera_automl.base_evaluation import EvaluationResult
from accelera.src.accelera_automl.base_evaluation import TrialSpecs
from accelera.src.accelera_automl.optimization.smac_optimizer import Optimizer
from accelera.src.accelera_automl.optimization.smac_optimizer import Trial


class FakeConfiguration(dict):
    def get_array(self):
        return [0.0]


def test_none_n_trials_runs_until_time_budget(monkeypatch):
    times = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(
        "accelera.src.accelera_automl.optimization.smac_optimizer.perf_counter",
        lambda: next(times),
    )

    config = FakeConfiguration(model_name="dummy")
    level = TrialSpecs(stage=0, sample_fraction=1.0, cv_folds=2, model_budget=1.0)
    optimizer = Optimizer(
        configspace={},
        evaluator=None,
        X=None,
        y=None,
        n_trials=None,
        time_budget=1.0,
        n_parallel=1,
        evaluation_level=[level],
        verbose=0,
    )
    monkeypatch.setattr(
        optimizer,
        "suggest_batch",
        lambda batch_size: [Trial(config=config, evaluation_level=level)],
    )
    monkeypatch.setattr(
        optimizer,
        "evaluate_single_config",
        lambda trial: EvaluationResult(
            model_name="dummy",
            params={},
            preprocessing="none",
            score=1.0,
            cost=0.0,
            duration=0.0,
            status="success",
            evaluation_level_stage=0,
            sample_fraction=1.0,
            cv_folds=2,
            model_budget=1.0,
        ),
    )

    result = optimizer.optimize()

    assert len(result.runhistory) == 1
    assert result.best_config == config


def test_none_n_trials_requires_time_budget():
    optimizer = Optimizer(
        configspace={},
        evaluator=None,
        X=None,
        y=None,
        n_trials=None,
        time_budget=None,
        verbose=0,
    )

    try:
        optimizer.optimize()
    except ValueError as exc:
        assert str(exc) == "n_trials=None requires a finite time_budget."
    else:
        raise AssertionError("Expected n_trials=None without time_budget to fail.")
