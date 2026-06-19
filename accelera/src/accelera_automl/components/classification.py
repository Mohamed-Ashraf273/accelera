from catboost import CatBoostClassifier
from ConfigSpace.conditions import AndConjunction
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class ClassificationComponent:
    def __init__(
        self, name, build_hyperparameters, build_conditions, build_estimator
    ):
        self.name = name
        self.build_hyperparameters = build_hyperparameters
        self.build_conditions = build_conditions
        self.build_estimator = build_estimator


def create_conditions(model_of_hyperparamerter, model_name, params):
    return [
        EqualsCondition(param, model_of_hyperparamerter, model_name)
        for param in params.values()
    ]


def create_logistic_regression_hyperparameters():
    return [
        UniformFloatHyperparameter(
            "logistic_regression:C", lower=1e-4, upper=1e4, log=True
        ),
        CategoricalHyperparameter(
            "logistic_regression:solver", choices=["lbfgs", "liblinear", "saga"]
        ),
        CategoricalHyperparameter(
            "logistic_regression:penalty", choices=["l2", "l1"]
        ),
        UniformIntegerHyperparameter(
            "logistic_regression:max_iter", lower=100, upper=2000
        ),
        CategoricalHyperparameter(
            "logistic_regression:class_weight", choices=["none", "balanced"]
        ),
    ]


def create_logistic_regression_conditions(data):
    params = data["params"]
    conditions = create_conditions(data["model_name"], "logistic_regression", params)
    conditions = [
        cond
        for cond in conditions
        if cond.child.name != "logistic_regression:penalty"
    ]
    conditions.append(
        AndConjunction(
            EqualsCondition(
                params["logistic_regression:penalty"],
                data["model_name"],
                "logistic_regression",
            ),
            InCondition(
                params["logistic_regression:penalty"],
                params["logistic_regression:solver"],
                ["liblinear", "saga"],
            ),
        )
    )
    return conditions


def build_logistic_regression(params, random_state, n_jobs, balance_classes):
    estimator_params = {
        "C": float(params["C"]),
        "solver": params["solver"],
        "penalty": params.get("penalty", "l2"),
        "max_iter": int(params["max_iter"]),
        "random_state": random_state,
    }
    class_weight = params.get("class_weight", "none")
    if balance_classes:
        estimator_params["class_weight"] = "balanced"
    elif class_weight != "none":
        estimator_params["class_weight"] = class_weight
    if estimator_params["solver"] == "lbfgs":
        estimator_params["penalty"] = "l2"
    return LogisticRegression(**estimator_params)


def create_random_forest_hyperparameters():
    return [
        UniformIntegerHyperparameter(
            "random_forest:n_estimators", lower=100, upper=1000
        ),
        UniformIntegerHyperparameter("random_forest:max_depth", lower=2, upper=64),
        UniformIntegerHyperparameter(
            "random_forest:min_samples_split", lower=2, upper=20
        ),
        UniformIntegerHyperparameter(
            "random_forest:min_samples_leaf", lower=1, upper=20
        ),
        CategoricalHyperparameter(
            "random_forest:criterion", choices=["gini", "entropy"]
        ),
        CategoricalHyperparameter(
            "random_forest:max_features", choices=["sqrt", "log2", None]
        ),
        CategoricalHyperparameter("random_forest:bootstrap", choices=[True, False]),
        UniformIntegerHyperparameter(
            "random_forest:max_leaf_nodes", lower=2, upper=512
        ),
        CategoricalHyperparameter(
            "random_forest:class_weight", choices=["none", "balanced"]
        ),
    ]


def create_random_forest_conditions(data):
    return create_conditions(data["model_name"], "random_forest", data["params"])


def build_random_forest(params, random_state, n_jobs, balance_classes):
    estimator_params = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "criterion": params["criterion"],
        "max_features": params.get("max_features"),
        "bootstrap": params.get("bootstrap", True),
        "max_leaf_nodes": int(params["max_leaf_nodes"]),
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    class_weight = params.get("class_weight", "none")
    if balance_classes:
        estimator_params["class_weight"] = "balanced"
    elif class_weight != "none":
        estimator_params["class_weight"] = class_weight
    return RandomForestClassifier(**estimator_params)


def create_extra_trees_hyperparameters():
    return [
        UniformIntegerHyperparameter(
            "extra_trees:n_estimators", lower=100, upper=1000
        ),
        UniformIntegerHyperparameter("extra_trees:max_depth", lower=2, upper=64),
        UniformIntegerHyperparameter(
            "extra_trees:min_samples_split", lower=2, upper=20
        ),
        UniformIntegerHyperparameter(
            "extra_trees:min_samples_leaf", lower=1, upper=20
        ),
        CategoricalHyperparameter(
            "extra_trees:criterion", choices=["gini", "entropy"]
        ),
        CategoricalHyperparameter(
            "extra_trees:max_features", choices=["sqrt", "log2", None]
        ),
        UniformIntegerHyperparameter(
            "extra_trees:max_leaf_nodes", lower=2, upper=512
        ),
        CategoricalHyperparameter(
            "extra_trees:class_weight", choices=["none", "balanced"]
        ),
    ]


def create_extra_trees_conditions(data):
    return create_conditions(data["model_name"], "extra_trees", data["params"])


def build_extra_trees(params, random_state, n_jobs, balance_classes):
    estimator_params = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "criterion": params["criterion"],
        "max_features": params.get("max_features"),
        "max_leaf_nodes": int(params["max_leaf_nodes"]),
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    class_weight = params.get("class_weight", "none")
    if balance_classes:
        estimator_params["class_weight"] = "balanced"
    elif class_weight != "none":
        estimator_params["class_weight"] = class_weight
    return ExtraTreesClassifier(**estimator_params)


def create_svc_hyperparameters():
    return [
        UniformFloatHyperparameter("svc:C", lower=1e-4, upper=1e4, log=True),
        CategoricalHyperparameter("svc:kernel", choices=["rbf", "poly", "sigmoid"]),
        UniformFloatHyperparameter("svc:gamma", lower=1e-5, upper=10.0, log=True),
        UniformIntegerHyperparameter("svc:degree", lower=2, upper=5),
        UniformFloatHyperparameter("svc:coef0", lower=-1.0, upper=1.0),
        CategoricalHyperparameter("svc:class_weight", choices=["none", "balanced"]),
    ]


def create_svc_conditions(data):
    params = data["params"]
    conditions = create_conditions(data["model_name"], "svc", params)
    conditions = [
        cond
        for cond in conditions
        if cond.child.name not in {"svc:degree", "svc:coef0"}
    ]
    conditions.extend(
        [
            AndConjunction(
                EqualsCondition(params["svc:degree"], data["model_name"], "svc"),
                EqualsCondition(params["svc:degree"], params["svc:kernel"], "poly"),
            ),
            AndConjunction(
                EqualsCondition(params["svc:coef0"], data["model_name"], "svc"),
                InCondition(
                    params["svc:coef0"], params["svc:kernel"], ["poly", "sigmoid"]
                ),
            ),
        ]
    )
    return conditions


def build_svc(params, random_state, n_jobs, balance_classes):
    estimator_params = {
        "C": float(params["C"]),
        "kernel": params.get("kernel", "rbf"),
        "gamma": float(params["gamma"]),
        "degree": int(params.get("degree", 3)),
        "coef0": float(params.get("coef0", 0.0)),
        "probability": True,
        "random_state": random_state,
    }
    class_weight = params.get("class_weight", "none")
    if balance_classes:
        estimator_params["class_weight"] = "balanced"
    elif class_weight != "none":
        estimator_params["class_weight"] = class_weight
    return SVC(**estimator_params)


def create_knn_hyperparameters():
    return [
        UniformIntegerHyperparameter("knn:n_neighbors", lower=1, upper=50),
        CategoricalHyperparameter("knn:weights", choices=["uniform", "distance"]),
        CategoricalHyperparameter("knn:p", choices=[1, 2]),
        CategoricalHyperparameter(
            "knn:algorithm", choices=["auto", "ball_tree", "kd_tree", "brute"]
        ),
        UniformIntegerHyperparameter("knn:leaf_size", lower=10, upper=100),
    ]


def create_knn_conditions(data):
    return create_conditions(data["model_name"], "knn", data["params"])


def build_knn(params, random_state, n_jobs, balance_classes):
    return KNeighborsClassifier(
        n_neighbors=int(params["n_neighbors"]),
        weights=params["weights"],
        p=int(params["p"]),
        algorithm=params.get("algorithm", "auto"),
        leaf_size=int(params.get("leaf_size", 30)),
    )


def create_gradient_boosting_hyperparameters():
    return [
        UniformIntegerHyperparameter(
            "gradient_boosting:n_estimators", lower=50, upper=500
        ),
        UniformFloatHyperparameter(
            "gradient_boosting:learning_rate", lower=1e-3, upper=0.5, log=True
        ),
        UniformIntegerHyperparameter(
            "gradient_boosting:max_depth", lower=1, upper=10
        ),
        UniformFloatHyperparameter(
            "gradient_boosting:subsample", lower=0.5, upper=1.0
        ),
        UniformIntegerHyperparameter(
            "gradient_boosting:max_leaf_nodes", lower=2, upper=128
        ),
    ]


def create_gradient_boosting_conditions(data):
    return create_conditions(data["model_name"], "gradient_boosting", data["params"])


def build_gradient_boosting(params, random_state, n_jobs, balance_classes):
    return GradientBoostingClassifier(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params["subsample"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        random_state=random_state,
    )


def create_lightgbm_hyperparameters():
    return [
        UniformIntegerHyperparameter("lightgbm:n_estimators", lower=100, upper=1200),
        UniformFloatHyperparameter(
            "lightgbm:learning_rate", lower=1e-3, upper=0.3, log=True
        ),
        UniformIntegerHyperparameter("lightgbm:num_leaves", lower=15, upper=255),
        UniformIntegerHyperparameter("lightgbm:max_depth", lower=-1, upper=16),
        UniformIntegerHyperparameter(
            "lightgbm:min_child_samples", lower=5, upper=100
        ),
        UniformFloatHyperparameter("lightgbm:subsample", lower=0.5, upper=1.0),
        UniformFloatHyperparameter(
            "lightgbm:colsample_bytree", lower=0.5, upper=1.0
        ),
        UniformFloatHyperparameter(
            "lightgbm:reg_alpha", lower=1e-8, upper=10.0, log=True
        ),
        UniformFloatHyperparameter(
            "lightgbm:reg_lambda", lower=1e-8, upper=10.0, log=True
        ),
    ]


def create_lightgbm_conditions(data):
    return create_conditions(data["model_name"], "lightgbm", data["params"])


def build_lightgbm(params, random_state, n_jobs, balance_classes):
    estimator_params = {
        "n_estimators": int(params["n_estimators"]),
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "max_depth": int(params["max_depth"]),
        "min_child_samples": int(params["min_child_samples"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": -1,
    }
    if balance_classes:
        estimator_params["class_weight"] = "balanced"
    return LGBMClassifier(**estimator_params)


def create_xgboost_hyperparameters():
    return [
        UniformIntegerHyperparameter("xgboost:n_estimators", lower=100, upper=1200),
        UniformFloatHyperparameter(
            "xgboost:learning_rate", lower=1e-3, upper=0.3, log=True
        ),
        UniformIntegerHyperparameter("xgboost:max_depth", lower=2, upper=12),
        UniformFloatHyperparameter(
            "xgboost:min_child_weight", lower=1e-2, upper=32.0, log=True
        ),
        UniformFloatHyperparameter("xgboost:subsample", lower=0.5, upper=1.0),
        UniformFloatHyperparameter("xgboost:colsample_bytree", lower=0.5, upper=1.0),
        UniformFloatHyperparameter(
            "xgboost:reg_alpha", lower=1e-8, upper=10.0, log=True
        ),
        UniformFloatHyperparameter(
            "xgboost:reg_lambda", lower=1e-8, upper=10.0, log=True
        ),
        UniformFloatHyperparameter(
            "xgboost:gamma", lower=1e-8, upper=10.0, log=True
        ),
    ]


def create_xgboost_conditions(data):
    return create_conditions(data["model_name"], "xgboost", data["params"])


def build_xgboost(params, random_state, n_jobs, balance_classes):
    estimator_params = {
        "n_estimators": int(params["n_estimators"]),
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "gamma": float(params["gamma"]),
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": 0,
        "eval_metric": "logloss",
    }
    if balance_classes:
        estimator_params["scale_pos_weight"] = 1.0
    return XGBClassifier(**estimator_params)


def create_catboost_hyperparameters():
    return [
        UniformIntegerHyperparameter("catboost:iterations", lower=100, upper=1200),
        UniformFloatHyperparameter(
            "catboost:learning_rate", lower=1e-3, upper=0.3, log=True
        ),
        UniformIntegerHyperparameter("catboost:depth", lower=4, upper=10),
        UniformFloatHyperparameter(
            "catboost:l2_leaf_reg", lower=1e-3, upper=30.0, log=True
        ),
        UniformFloatHyperparameter(
            "catboost:random_strength", lower=1e-3, upper=10.0, log=True
        ),
        UniformFloatHyperparameter(
            "catboost:bagging_temperature", lower=0.0, upper=10.0
        ),
    ]


def create_catboost_conditions(data):
    return create_conditions(data["model_name"], "catboost", data["params"])


def build_catboost(params, random_state, n_jobs, balance_classes):
    estimator_params = {
        "iterations": int(params["iterations"]),
        "learning_rate": float(params["learning_rate"]),
        "depth": int(params["depth"]),
        "l2_leaf_reg": float(params["l2_leaf_reg"]),
        "random_strength": float(params["random_strength"]),
        "bagging_temperature": float(params["bagging_temperature"]),
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
    }
    if n_jobs is not None:
        estimator_params["thread_count"] = int(n_jobs)
    if balance_classes:
        estimator_params["auto_class_weights"] = "Balanced"
    return CatBoostClassifier(**estimator_params)


def create_hist_gradient_boosting_hyperparameters():
    return [
        UniformFloatHyperparameter(
            "hist_gradient_boosting:learning_rate", lower=1e-3, upper=0.5, log=True
        ),
        UniformIntegerHyperparameter(
            "hist_gradient_boosting:max_iter", lower=50, upper=500
        ),
        UniformIntegerHyperparameter(
            "hist_gradient_boosting:max_depth", lower=2, upper=32
        ),
        UniformIntegerHyperparameter(
            "hist_gradient_boosting:min_samples_leaf", lower=10, upper=100
        ),
        UniformFloatHyperparameter(
            "hist_gradient_boosting:l2_regularization",
            lower=1e-10,
            upper=1.0,
            log=True,
        ),
        UniformIntegerHyperparameter(
            "hist_gradient_boosting:max_bins", lower=32, upper=255
        ),
        CategoricalHyperparameter(
            "hist_gradient_boosting:early_stopping", choices=[False, True]
        ),
    ]


def create_hist_gradient_boosting_conditions(data):
    return create_conditions(
        data["model_name"], "hist_gradient_boosting", data["params"]
    )


def build_hist_gradient_boosting(params, random_state, n_jobs, balance_classes):
    return HistGradientBoostingClassifier(
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        max_bins=int(params["max_bins"]),
        early_stopping=bool(params["early_stopping"]),
        random_state=random_state,
    )


def create_adaboost_hyperparameters():
    return [
        UniformIntegerHyperparameter("adaboost:n_estimators", lower=25, upper=500),
        UniformFloatHyperparameter(
            "adaboost:learning_rate", lower=1e-3, upper=2.0, log=True
        ),
    ]


def create_adaboost_conditions(data):
    return create_conditions(data["model_name"], "adaboost", data["params"])


def build_adaboost(params, random_state, n_jobs, balance_classes):
    return AdaBoostClassifier(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        random_state=random_state,
    )


def create_gaussian_nb_hyperparameters():
    return [
        UniformFloatHyperparameter(
            "gaussian_nb:var_smoothing", lower=1e-12, upper=1e-6, log=True
        ),
    ]


def create_gaussian_nb_conditions(data):
    return create_conditions(data["model_name"], "gaussian_nb", data["params"])


def build_gaussian_nb(params, random_state, n_jobs, balance_classes):
    return GaussianNB(var_smoothing=float(params["var_smoothing"]))


def create_lda_hyperparameters():
    return [
        CategoricalHyperparameter("lda:solver", choices=["svd", "lsqr", "eigen"]),
        CategoricalHyperparameter("lda:shrinkage", choices=[None, "auto"]),
    ]


def create_lda_conditions(data):
    params = data["params"]
    conditions = create_conditions(data["model_name"], "lda", params)
    conditions = [cond for cond in conditions if cond.child.name != "lda:shrinkage"]
    conditions.append(
        AndConjunction(
            EqualsCondition(params["lda:shrinkage"], data["model_name"], "lda"),
            InCondition(
                params["lda:shrinkage"], params["lda:solver"], ["lsqr", "eigen"]
            ),
        )
    )
    return conditions


def build_lda(params, random_state, n_jobs, balance_classes):
    estimator_params = {"solver": params["solver"]}
    if params["solver"] in {"lsqr", "eigen"}:
        estimator_params["shrinkage"] = params.get("shrinkage")
    return LinearDiscriminantAnalysis(**estimator_params)


def create_decision_tree_hyperparameters():
    return [
        CategoricalHyperparameter(
            "decision_tree:criterion", choices=["gini", "entropy"]
        ),
        UniformIntegerHyperparameter("decision_tree:max_depth", lower=1, upper=64),
        UniformIntegerHyperparameter(
            "decision_tree:min_samples_split", lower=2, upper=20
        ),
        UniformIntegerHyperparameter(
            "decision_tree:min_samples_leaf", lower=1, upper=20
        ),
        CategoricalHyperparameter(
            "decision_tree:max_features", choices=["sqrt", "log2", None]
        ),
        CategoricalHyperparameter(
            "decision_tree:class_weight", choices=["none", "balanced"]
        ),
    ]


def create_decision_tree_conditions(data):
    return create_conditions(data["model_name"], "decision_tree", data["params"])


def build_decision_tree(params, random_state, n_jobs, balance_classes):
    estimator_params = {
        "criterion": params["criterion"],
        "max_depth": int(params["max_depth"]),
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "max_features": params.get("max_features"),
        "random_state": random_state,
    }
    class_weight = params.get("class_weight", "none")
    if balance_classes:
        estimator_params["class_weight"] = "balanced"
    elif class_weight != "none":
        estimator_params["class_weight"] = class_weight
    return DecisionTreeClassifier(**estimator_params)


def create_sgd_hyperparameters():
    return [
        CategoricalHyperparameter(
            "sgd:loss", choices=["hinge", "log_loss", "modified_huber"]
        ),
        CategoricalHyperparameter("sgd:penalty", choices=["l2", "l1", "elasticnet"]),
        UniformFloatHyperparameter("sgd:alpha", lower=1e-6, upper=1e-1, log=True),
        CategoricalHyperparameter(
            "sgd:learning_rate", choices=["optimal", "invscaling", "adaptive"]
        ),
        UniformFloatHyperparameter("sgd:eta0", lower=1e-4, upper=1e-1, log=True),
        CategoricalHyperparameter("sgd:average", choices=[False, True]),
        CategoricalHyperparameter("sgd:class_weight", choices=["none", "balanced"]),
    ]


def create_sgd_conditions(data):
    return create_conditions(data["model_name"], "sgd", data["params"])


def build_sgd(params, random_state, n_jobs, balance_classes):
    estimator_params = {
        "loss": params["loss"],
        "penalty": params["penalty"],
        "alpha": float(params["alpha"]),
        "learning_rate": params["learning_rate"],
        "eta0": float(params["eta0"]),
        "average": bool(params["average"]),
        "random_state": random_state,
    }
    class_weight = params.get("class_weight", "none")
    if balance_classes:
        estimator_params["class_weight"] = "balanced"
    elif class_weight != "none":
        estimator_params["class_weight"] = class_weight
    return SGDClassifier(**estimator_params)


def create_passive_aggressive_hyperparameters():
    return [
        UniformFloatHyperparameter(
            "passive_aggressive:C", lower=1e-4, upper=10.0, log=True
        ),
        CategoricalHyperparameter(
            "passive_aggressive:loss", choices=["hinge", "squared_hinge"]
        ),
        CategoricalHyperparameter(
            "passive_aggressive:average", choices=[False, True]
        ),
    ]


def create_passive_aggressive_conditions(data):
    return create_conditions(
        data["model_name"], "passive_aggressive", data["params"]
    )


def build_passive_aggressive(params, random_state, n_jobs, balance_classes):
    return PassiveAggressiveClassifier(
        C=float(params["C"]),
        loss=params["loss"],
        average=bool(params["average"]),
        random_state=random_state,
        class_weight="balanced" if balance_classes else None,
    )


def create_qda_hyperparameters():
    return [
        UniformFloatHyperparameter("qda:reg_param", lower=0.0, upper=1.0),
    ]


def create_qda_conditions(data):
    return create_conditions(data["model_name"], "qda", data["params"])


def build_qda(params, random_state, n_jobs, balance_classes):
    return QuadraticDiscriminantAnalysis(reg_param=float(params["reg_param"]))


def create_liblinear_svc_hyperparameters():
    return [
        UniformFloatHyperparameter(
            "liblinear_svc:C", lower=1e-4, upper=1e4, log=True
        ),
        CategoricalHyperparameter(
            "liblinear_svc:loss", choices=["hinge", "squared_hinge"]
        ),
        CategoricalHyperparameter(
            "liblinear_svc:class_weight", choices=["none", "balanced"]
        ),
    ]


def create_liblinear_svc_conditions(data):
    return create_conditions(data["model_name"], "liblinear_svc", data["params"])


def build_liblinear_svc(params, random_state, n_jobs, balance_classes):
    estimator_params = {
        "C": float(params["C"]),
        "loss": params["loss"],
        "random_state": random_state,
    }
    class_weight = params.get("class_weight", "none")
    if balance_classes:
        estimator_params["class_weight"] = "balanced"
    elif class_weight != "none":
        estimator_params["class_weight"] = class_weight
    return LinearSVC(**estimator_params)


def create_bernoulli_nb_hyperparameters():
    return [
        UniformFloatHyperparameter(
            "bernoulli_nb:alpha", lower=1e-3, upper=100.0, log=True
        ),
        CategoricalHyperparameter("bernoulli_nb:fit_prior", choices=[False, True]),
    ]


def create_bernoulli_nb_conditions(data):
    return create_conditions(data["model_name"], "bernoulli_nb", data["params"])


def build_bernoulli_nb(params, random_state, n_jobs, balance_classes):
    return BernoulliNB(
        alpha=float(params["alpha"]),
        fit_prior=bool(params["fit_prior"]),
    )


def create_multinomial_nb_hyperparameters():
    return [
        UniformFloatHyperparameter(
            "multinomial_nb:alpha", lower=1e-3, upper=100.0, log=True
        ),
        CategoricalHyperparameter("multinomial_nb:fit_prior", choices=[False, True]),
    ]


def create_multinomial_nb_conditions(data):
    return create_conditions(data["model_name"], "multinomial_nb", data["params"])


def build_multinomial_nb(params, random_state, n_jobs, balance_classes):
    return MultinomialNB(
        alpha=float(params["alpha"]),
        fit_prior=bool(params["fit_prior"]),
    )


def create_mlp_hyperparameters():
    return [
        UniformIntegerHyperparameter("mlp:hidden_layer_sizes", lower=32, upper=256),
        UniformFloatHyperparameter("mlp:alpha", lower=1e-6, upper=1e-1, log=True),
        UniformFloatHyperparameter(
            "mlp:learning_rate_init", lower=1e-4, upper=1e-1, log=True
        ),
        CategoricalHyperparameter(
            "mlp:activation", choices=["relu", "tanh", "logistic"]
        ),
        CategoricalHyperparameter("mlp:solver", choices=["adam", "sgd"]),
    ]


def create_mlp_conditions(data):
    return create_conditions(data["model_name"], "mlp", data["params"])


def build_mlp(params, random_state, n_jobs, balance_classes):
    return MLPClassifier(
        hidden_layer_sizes=(int(params["hidden_layer_sizes"]),),
        alpha=float(params["alpha"]),
        learning_rate_init=float(params["learning_rate_init"]),
        activation=params["activation"],
        solver=params["solver"],
        max_iter=400,
        early_stopping=True,
        random_state=random_state,
    )


def create_gaussian_process_hyperparameters():
    return [
        UniformFloatHyperparameter(
            "gaussian_process:length_scale", lower=1e-2, upper=1e2, log=True
        ),
        UniformIntegerHyperparameter(
            "gaussian_process:max_iter_predict", lower=20, upper=200
        ),
    ]


def create_gaussian_process_conditions(data):
    return create_conditions(data["model_name"], "gaussian_process", data["params"])


def build_gaussian_process(params, random_state, n_jobs, balance_classes):
    return GaussianProcessClassifier(
        kernel=1.0 * RBF(length_scale=float(params["length_scale"])),
        max_iter_predict=int(params["max_iter_predict"]),
        random_state=random_state,
    )


def create_ridge_classifier_hyperparameters():
    return [
        UniformFloatHyperparameter(
            "ridge_classifier:alpha", lower=1e-4, upper=1e3, log=True
        ),
        CategoricalHyperparameter(
            "ridge_classifier:class_weight", choices=["none", "balanced"]
        ),
        CategoricalHyperparameter(
            "ridge_classifier:solver",
            choices=["auto", "svd", "cholesky", "lsqr", "sag"],
        ),
    ]


def create_ridge_classifier_conditions(data):
    return create_conditions(data["model_name"], "ridge_classifier", data["params"])


def build_ridge_classifier(params, random_state, n_jobs, balance_classes):
    estimator_params = {
        "alpha": float(params["alpha"]),
        "solver": params["solver"],
        "random_state": random_state,
    }
    class_weight = params.get("class_weight", "none")
    if balance_classes:
        estimator_params["class_weight"] = "balanced"
    elif class_weight != "none":
        estimator_params["class_weight"] = class_weight
    return RidgeClassifier(**estimator_params)


CLASSIFICATION_COMPONENTS = {
    "adaboost": ClassificationComponent(
        "adaboost",
        create_adaboost_hyperparameters,
        create_adaboost_conditions,
        build_adaboost,
    ),
    "bernoulli_nb": ClassificationComponent(
        "bernoulli_nb",
        create_bernoulli_nb_hyperparameters,
        create_bernoulli_nb_conditions,
        build_bernoulli_nb,
    ),
    "catboost": ClassificationComponent(
        "catboost",
        create_catboost_hyperparameters,
        create_catboost_conditions,
        build_catboost,
    ),
    "decision_tree": ClassificationComponent(
        "decision_tree",
        create_decision_tree_hyperparameters,
        create_decision_tree_conditions,
        build_decision_tree,
    ),
    "extra_trees": ClassificationComponent(
        "extra_trees",
        create_extra_trees_hyperparameters,
        create_extra_trees_conditions,
        build_extra_trees,
    ),
    "gaussian_nb": ClassificationComponent(
        "gaussian_nb",
        create_gaussian_nb_hyperparameters,
        create_gaussian_nb_conditions,
        build_gaussian_nb,
    ),
    "gradient_boosting": ClassificationComponent(
        "gradient_boosting",
        create_gradient_boosting_hyperparameters,
        create_gradient_boosting_conditions,
        build_gradient_boosting,
    ),
    "hist_gradient_boosting": ClassificationComponent(
        "hist_gradient_boosting",
        create_hist_gradient_boosting_hyperparameters,
        create_hist_gradient_boosting_conditions,
        build_hist_gradient_boosting,
    ),
    "knn": ClassificationComponent(
        "knn", create_knn_hyperparameters, create_knn_conditions, build_knn
    ),
    "lda": ClassificationComponent(
        "lda", create_lda_hyperparameters, create_lda_conditions, build_lda
    ),
    "lightgbm": ClassificationComponent(
        "lightgbm",
        create_lightgbm_hyperparameters,
        create_lightgbm_conditions,
        build_lightgbm,
    ),
    "liblinear_svc": ClassificationComponent(
        "liblinear_svc",
        create_liblinear_svc_hyperparameters,
        create_liblinear_svc_conditions,
        build_liblinear_svc,
    ),
    "logistic_regression": ClassificationComponent(
        "logistic_regression",
        create_logistic_regression_hyperparameters,
        create_logistic_regression_conditions,
        build_logistic_regression,
    ),
    "multinomial_nb": ClassificationComponent(
        "multinomial_nb",
        create_multinomial_nb_hyperparameters,
        create_multinomial_nb_conditions,
        build_multinomial_nb,
    ),
    "mlp": ClassificationComponent(
        "mlp", create_mlp_hyperparameters, create_mlp_conditions, build_mlp
    ),
    "passive_aggressive": ClassificationComponent(
        "passive_aggressive",
        create_passive_aggressive_hyperparameters,
        create_passive_aggressive_conditions,
        build_passive_aggressive,
    ),
    "qda": ClassificationComponent(
        "qda", create_qda_hyperparameters, create_qda_conditions, build_qda
    ),
    "gaussian_process": ClassificationComponent(
        "gaussian_process",
        create_gaussian_process_hyperparameters,
        create_gaussian_process_conditions,
        build_gaussian_process,
    ),
    "random_forest": ClassificationComponent(
        "random_forest",
        create_random_forest_hyperparameters,
        create_random_forest_conditions,
        build_random_forest,
    ),
    "ridge_classifier": ClassificationComponent(
        "ridge_classifier",
        create_ridge_classifier_hyperparameters,
        create_ridge_classifier_conditions,
        build_ridge_classifier,
    ),
    "sgd": ClassificationComponent(
        "sgd", create_sgd_hyperparameters, create_sgd_conditions, build_sgd
    ),
    "svc": ClassificationComponent(
        "svc", create_svc_hyperparameters, create_svc_conditions, build_svc
    ),
    "xgboost": ClassificationComponent(
        "xgboost",
        create_xgboost_hyperparameters,
        create_xgboost_conditions,
        build_xgboost,
    ),
}


def get_classification_components(allowed_models=None):
    if allowed_models is None:
        return dict(CLASSIFICATION_COMPONENTS)

    selected_models = sorted(set(allowed_models))
    unknown = sorted(set(selected_models) - set(CLASSIFICATION_COMPONENTS))
    if unknown:
        raise ValueError(f"unknown classification models requested: {unknown}")

    if not selected_models:
        raise ValueError("at least one classification model must be selected.")

    return {name: CLASSIFICATION_COMPONENTS[name] for name in selected_models}
