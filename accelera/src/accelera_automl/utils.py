from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,f1_score,log_loss,precision_score,recall_score,roc_auc_score
import numpy as np

def score_predictions(classes,scoring,y_true,predictions):
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
            return float(
                recall_score(y_true, y_pred, average=average, zero_division=0)
            )
        if scoring == "recall_macro":
            return float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            )
        if scoring == "recall_micro":
            return float(
                recall_score(y_true, y_pred, average="micro", zero_division=0)
            )
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


def log_forward_selection_step(
        step,
        selected_names,
        score,
        improvement,
    ):
        if improvement is None:
            print(
                f"forward selection step {step}: "
                f"selected {selected_names[-1]} score={score:.6f}"
            )
            return
        print(
            f"forward selection step {step}: "
            f"added {selected_names[-1]} score={score:.6f} "
            f"improvement={improvement:.6f}"
        )


def log_ensemble_structure(base_model_names,meta_modelname,score,include_original_features_in_meta):
        print("stacked ensemble summary")
        print(f"selected base models: {', '.join(base_model_names)}")
        print(f"meta model: {meta_modelname}")
        print(
            f"forward selection score: {score:.6f} "
            f"(uses original features={include_original_features_in_meta})"
        )