import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    log_loss,
)


CLASS_ORDER = ["H", "D", "A"]


def multiclass_brier_score(y_true: pd.Series, proba: np.ndarray, class_order=None) -> float:
    if class_order is None:
        class_order = CLASS_ORDER

    y_true_onehot = np.zeros((len(y_true), len(class_order)))
    class_to_idx = {c: i for i, c in enumerate(class_order)}

    for row_idx, label in enumerate(y_true):
        y_true_onehot[row_idx, class_to_idx[label]] = 1.0

    return np.mean(np.sum((proba - y_true_onehot) ** 2, axis=1))


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str) -> pd.DataFrame:
    row = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "draw_recall": f1_score((y_true == "D").astype(int), (y_pred == "D").astype(int)),
        "log_loss": log_loss(y_true, y_proba, labels=CLASS_ORDER),
        "brier_multiclass": multiclass_brier_score(y_true, y_proba, class_order=CLASS_ORDER),
    }
    return pd.DataFrame([row])


def build_confusion(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    return pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)


def evaluate_model_dict(y_true: pd.Series, predictions: dict) -> pd.DataFrame:
    rows = []
    for model_name, pred_dict in predictions.items():
        row_df = evaluate_predictions(
            y_true=y_true,
            y_pred=pred_dict["pred"],
            y_proba=pred_dict["proba"],
            model_name=model_name,
        )
        rows.append(row_df)

    return pd.concat(rows, ignore_index=True).sort_values("log_loss")