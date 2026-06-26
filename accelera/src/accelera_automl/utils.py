import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def score_predictions(classes, scoring, y_true, predictions):
    y_pred = classes[np.argmax(predictions, axis=1)]
    scoring = scoring

    if scoring == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if scoring == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if scoring == "f1":
        average = "binary" if len(classes) == 2 else "macro"
        return float(f1_score(y_true, y_pred, average=average))
    if scoring == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro"))
    if scoring == "f1_micro":
        return float(f1_score(y_true, y_pred, average="micro"))
    if scoring == "f1_weighted":
        return float(f1_score(y_true, y_pred, average="weighted"))
    if scoring == "precision":
        average = "binary" if len(classes) == 2 else "macro"
        return float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        )
    if scoring == "precision_macro":
        return float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
    if scoring == "precision_micro":
        return float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        )
    if scoring == "precision_weighted":
        return float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
    if scoring == "recall":
        average = "binary" if len(classes) == 2 else "macro"
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))
    if scoring == "recall_macro":
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    if scoring == "recall_micro":
        return float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    if scoring == "recall_weighted":
        return float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
    if scoring == "roc_auc" and predictions.shape[1] == 2:
        return float(roc_auc_score(y_true, predictions[:, 1]))
    if scoring == "average_precision" and predictions.shape[1] == 2:
        return float(average_precision_score(y_true, predictions[:, 1]))
    if scoring in {"neg_log_loss", "log_loss"}:
        return float(-log_loss(y_true, predictions, labels=classes))
    return float(accuracy_score(y_true, y_pred))


def log_forward_selection_step(selected_names, score):
    print(
        f"Forward selection selected {selected_names} with validation score {score}."
    )


def log_ensemble_structure(base_model_names, meta_model_name):
    print(
        "Stacked ensemble base models: "
        f"{base_model_names}; meta model: {meta_model_name}."
    )
