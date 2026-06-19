from ConfigSpace.configuration_space import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from .components import get_classification_components
from .components import get_regression_components


def return_classification_config_space(allowed_models=None):
    config_space = ConfigurationSpace()
    components = get_classification_components(allowed_models=allowed_models)
    chosen_models = sorted(components)  # sort by model_name

    model_name = CategoricalHyperparameter(
        "model_name", choices=chosen_models
    )  # model_name as hyperparameter with models choices
    config_space.add(model_name)

    for name in chosen_models:
        component = components[name]
        hyp = component.build_hyperparameters()
        config_space.add(hyp)
        params = {param.name: param for param in hyp}
        conditions = component.build_conditions(
            {"model_name": model_name, "params": params}
        )
        if conditions:
            config_space.add(conditions)

    return config_space


def return_regression_config_space(allowed_models=None):
    config_space = ConfigurationSpace()
    components = get_regression_components(allowed_models=allowed_models)
    chosen_models = sorted(components)  # sort by model_name

    model_name = CategoricalHyperparameter(
        "model_name", choices=chosen_models
    )  # model_name as hyperparameter with models choices
    config_space.add(model_name)

    for name in chosen_models:
        component = components[name]
        hyp = component.build_hyperparameters()
        config_space.add(hyp)
        params = {param.name: param for param in hyp}
        conditions = component.build_conditions(
            {"model_name": model_name, "params": params}
        )
        if conditions:
            config_space.add(conditions)

    return config_space


def sample_classification_config(configspace):
    return (
        configspace.sample_configuration()
    )  # sample randomly one model with its hyperparameters


def configuration_space_to_dict(config):
    raw = dict(config)
    model_name = raw["model_name"]
    pref = f"{model_name}:"

    params = {
        key[len(pref) :]: value for key, value in raw.items() if key.startswith(pref)
    }
    return {
        "model_name": model_name,
        "params": params,
    }


def dict_to_configuration_space(config_dict, config_space):
    name = config_dict["model_name"]
    params = config_dict.get("params", {})

    config = {"model_name": name}
    for key, value in params.items():
        config[f"{name}:{key}"] = value

    return Configuration(config_space, values=config)
