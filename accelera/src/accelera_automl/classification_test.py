import numpy as np
from sklearn.dummy import DummyClassifier

from accelera.src.accelera_automl.classification import AutoMLClassifier
from accelera.src.accelera_automl.core.automl import AutoMLResult


def test_classifier_runs_with_all_options(monkeypatch):
    captured = {}

    def fake_search(engine, X, y):
        captured["engine"] = engine
        model = DummyClassifier(strategy="prior").fit(X, y)
        return AutoMLResult(
            best_model=model,
            best_score=1.0,
            leaderboard=[],
            ensemble=None,
            final_model=model,
            runhistory=[],
        )

    monkeypatch.setattr(
        "accelera.src.accelera_automl.core.automl.AutoMLEngine.search",
        fake_search,
    )

    automl = AutoMLClassifier(
        time_budget=2.0,
        per_run_time_limit=0.5,
        n_trials=None,
        cv=2,
        scoring="roc_auc",
        random_state=7,
        use_ensemble=False,
        ensemble_voting_size=3,
        ensemble_strategy="voting",
        stacked_base_size=2,
        stacked_bagging_n_estimators=2,
        stacked_include_original_features_in_meta=True,
        search_n_parallel=1,
        stack_n_jobs=1,
        inner_n_jobs=1,
        disable_evaluation_timeout=True,
        balance_classes=True,
        allowed_models=["decision_tree"],
        use_meta_learning=False,
        meta_learning_top_datasets=2,
        meta_learning_top_configs_per_dataset=2,
        max_meta_learning_warmstarts=2,
        candidate_pool_size=16,
        n_initial_points=1,
        verbose=0,
    )
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    automl.fit(X, y)
    predictions = automl.predict(X)
    probabilities = automl.predict_proba(X)

    engine = captured["engine"]
    assert predictions.shape == (4,)
    assert probabilities.shape == (4, 2)
    assert automl.is_fitted
    assert engine.task == "classification"
    assert engine.time_budget == 2.0
    assert engine.per_run_time_limit == 0.5
    assert engine.n_trials is None
    assert engine.cv == 2
    assert engine.scoring == "roc_auc"
    assert engine.random_state == 7
    assert not engine.ensemble
    assert engine.ensemble_voting_size == 3
    assert engine.ensemble_strategy == "voting"
    assert engine.stacked_base_size == 2
    assert engine.stacked_bagging_n_estimators == 2
    assert engine.stacked_include_original_features_in_meta
    assert engine.search_n_parallel == 1
    assert engine.stack_n_jobs == 1
    assert engine.inner_n_jobs == 1
    assert engine.disable_evaluation_timeout
    assert engine.balance_classes
    assert engine.allowed_models == ["decision_tree"]
    assert not engine.use_meta_learning
    assert engine.meta_learning_top_datasets == 2
    assert engine.meta_learning_top_configs_per_dataset == 2
    assert engine.max_meta_learning_warmstarts == 2
    assert engine.candidate_pool_size == 16
    assert engine.n_initial_points == 1
    assert engine.verbose == 0
