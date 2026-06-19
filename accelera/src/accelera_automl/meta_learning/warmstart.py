import json
from pathlib import Path

import numpy as np

from ..configspace_search_space import dict_to_configuration_space


def get_meta_learning_warmstarts(
    *,
    task="classification",
    y=None,
    metafeatures=None,
    configspace=None,
    scoring=None,
    allowed_models=None,
    top_datasets=5,
    top_configs_per_dataset=3,
    max_warmstarts=10,
):
    metadata_directory = "json" if task == "classification" else "json_regression"
    data_path = resolve_path(
        task=task,
        y=y,
        scoring=scoring,
        meta_learning_root=(
            Path(__file__).resolve().parents[1]
            / "meta_learning_data"
            / metadata_directory
        ),
    )
    if data_path is None or not data_path.exists():
        return []

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", [])
    if not datasets:
        return []

    ranked_datasets = sorted(
        datasets,
        key=lambda dataset: return_similarity_distance(
            metafeatures,
            dataset.get("metafeatures", {}),
        ),
    )

    allowed = set(allowed_models) if allowed_models is not None else None
    selected_configs = []
    seen_signatures = set()

    for dataset in ranked_datasets[:top_datasets]:
        candidates = dataset.get("candidates", [])[:top_configs_per_dataset]
        for candidate in candidates:
            if allowed is not None and candidate["model_name"] not in allowed:
                continue

            signature = (
                candidate["model_name"],
                tuple(sorted(candidate["params"].items())),
            )
            if signature in seen_signatures:
                continue

            try:
                params = normalize_candidate_params(
                    candidate["model_name"],
                    candidate["params"],
                    configspace,
                )
                config = dict_to_configuration_space(
                    {
                        "model_name": candidate["model_name"],
                        "params": params,
                    },
                    configspace,
                )
            except Exception:
                continue

            selected_configs.append(config)
            seen_signatures.add(signature)
            if len(selected_configs) >= max_warmstarts:
                return selected_configs

    return selected_configs


def normalize_candidate_params(model_name, params, configspace):
    """Adapt exported configurations to the current ConfigSpace schema."""
    parameter_aliases = {
        ("ard_regression", "n_iter"): "max_iter",
    }
    value_aliases = {
        "criterion": {
            "mae": "absolute_error",
            "mse": "squared_error",
        },
        "loss": {
            "least_absolute_deviation": "absolute_error",
            "least_squares": "squared_error",
        },
    }

    normalized = {}
    for original_name, original_value in params.items():
        name = parameter_aliases.get((model_name, original_name), original_name)
        full_name = f"{model_name}:{name}"
        try:
            hyperparameter = configspace[full_name]
        except KeyError:
            continue

        value = value_aliases.get(name, {}).get(original_value, original_value)
        if hasattr(hyperparameter, "lower") and hasattr(hyperparameter, "upper"):
            value = min(max(value, hyperparameter.lower), hyperparameter.upper)
            if isinstance(hyperparameter.lower, int):
                value = int(value)
        elif (
            hasattr(hyperparameter, "choices")
            and value not in hyperparameter.choices
        ):
            continue

        normalized[name] = value
    return normalized


def resolve_path(
    *,
    task,
    y,
    scoring,
    meta_learning_root,
):
    candidate_names = return_path_names(task=task, y=y, scoring=scoring)
    for name in candidate_names:
        path = meta_learning_root / name
        if path.exists():
            return path
    return None


def return_path_names(*, task, y, scoring):
    metric_name = scoring or ("accuracy" if task == "classification" else "r2")
    if task == "classification":
        n_classes = len(np.unique(np.asarray(y)))
        task_suffix = "binary" if n_classes <= 2 else "multiclass"
        return [
            f"{metric_name}_{task_suffix}.classification_dense.json",
            f"{metric_name}.classification_dense.json",
        ]
    if task == "regression":
        return [
            f"{metric_name}_regression_dense.json",
        ]
    return []


def return_similarity_distance(current, historical):
    common_keys = [
        key
        for key in current
        if key in historical and isinstance(historical[key], (int, float))
    ]
    if not common_keys:
        return float("inf")

    diffs = []
    for key in common_keys:
        current_value = float(current[key])
        historical_value = float(historical[key])
        scale = max(1.0, abs(current_value), abs(historical_value))
        diffs.append(((current_value - historical_value) / scale) ** 2)
    return float(np.sqrt(np.sum(diffs)))
