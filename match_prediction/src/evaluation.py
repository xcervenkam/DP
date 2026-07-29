import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


RESULT_CLASSES = np.array(["H", "D", "A"])


def evaluate_multiclass(
    y_true,
    probabilities,
    labels: np.ndarray = RESULT_CLASSES,
) -> dict[str, float]:
    """Evaluate H/D/A probability predictions."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = labels[np.argmax(probabilities, axis=1)]
    observed = (y_true[:, None] == labels[None, :]).astype(float)

    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)

    return {
        "accuracy": accuracy_score(y_true, predicted),
        "macro_f1": f1_score(y_true, predicted, labels=labels, average="macro"),
        "home_win_recall": recall_score(y_true, predicted, labels=["H"], average="macro", zero_division=0),
        "draw_recall": recall_score(y_true == "D", predicted == "D"),
        "away_win_recall": recall_score(y_true, predicted, labels=["A"], average="macro", zero_division=0),
        "log_loss": -np.mean(np.sum(observed * np.log(clipped), axis=1)),
        "brier_score": np.mean(np.sum((probabilities - observed) ** 2, axis=1)),
        "n_matches": len(y_true),
    }


def evaluate_binary(y_true, probability_home_win) -> dict[str, float]:
    """Evaluate home-win versus home-non-win probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    probability_home_win = np.asarray(probability_home_win, dtype=float)
    predicted = (probability_home_win >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "precision_home_win": precision_score(y_true, predicted, zero_division=0),
        "f1_home_win": f1_score(y_true, predicted, zero_division=0),
        "home_win_recall": recall_score(y_true, predicted, zero_division=0),
        "specificity_home_non_win": specificity,
        "log_loss": log_loss(y_true, probability_home_win, labels=[0, 1]),
        "brier_score": np.mean((probability_home_win - y_true) ** 2),
        "n_matches": len(y_true),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def wilson_accuracy_interval(
    correct: int,
    n_matches: int,
    z_value: float = 1.96,
) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial accuracy proportion."""
    if n_matches <= 0:
        return np.nan, np.nan
    proportion = correct / n_matches
    denominator = 1 + z_value**2 / n_matches
    centre = (proportion + z_value**2 / (2 * n_matches)) / denominator
    half_width = (
        z_value
        * np.sqrt(
            proportion * (1 - proportion) / n_matches
            + z_value**2 / (4 * n_matches**2)
        )
        / denominator
    )
    return centre - half_width, centre + half_width
