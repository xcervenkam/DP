import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROBABILITY_METRICS = {"log_loss", "brier_score", "multiclass_brier"}


def metric_higher_is_better(metric_name: str) -> bool:
    """
    Return whether a metric should be maximized instead of minimized.
    """
    return metric_name not in PROBABILITY_METRICS


def safe_predict_proba(estimator, X: pd.DataFrame) -> tuple[np.ndarray | None, list | None]:
    """
    Safely obtain class probabilities from an estimator when available.
    """
    if not hasattr(estimator, "predict_proba"):
        return None, None

    try:
        y_proba = estimator.predict_proba(X)
    except Exception:
        return None, None

    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "named_steps"):
        final_model = estimator.named_steps.get("model")
        classes = getattr(final_model, "classes_", None)

    return y_proba, list(classes) if classes is not None else None


def _score_probability_predictions(
    y_true: pd.Series,
    y_proba: np.ndarray | None,
    proba_classes: list | None,
) -> dict:
    """
    Compute probability-quality metrics when class probabilities are available.
    """
    default_metrics = {
        "log_loss": np.nan,
        "brier_score": np.nan,
        "multiclass_brier": np.nan,
    }

    if y_proba is None or proba_classes is None:
        return default_metrics

    y_true_array = pd.Series(y_true).to_numpy()
    classes = list(proba_classes)
    clipped = np.clip(np.asarray(y_proba, dtype=float), 1e-9, 1 - 1e-9)

    if len(classes) == 2:
        positive_class = 1 if 1 in classes else classes[-1]
        positive_index = classes.index(positive_class)
        positive_probs = clipped[:, positive_index]
        binary_target = (y_true_array == positive_class).astype(int)

        return {
            "log_loss": log_loss(binary_target, positive_probs, labels=[0, 1]),
            "brier_score": brier_score_loss(binary_target, positive_probs),
            "multiclass_brier": np.nan,
        }

    log_loss_value = log_loss(y_true_array, clipped, labels=classes)
    y_true_matrix = np.column_stack([(y_true_array == cls).astype(float) for cls in classes])
    multiclass_brier = np.mean(np.sum((y_true_matrix - clipped) ** 2, axis=1))

    return {
        "log_loss": log_loss_value,
        "brier_score": np.nan,
        "multiclass_brier": multiclass_brier,
    }


def score_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    proba_classes: list | None = None,
) -> dict:
    """
    Compute classification metrics and, when available, probability metrics.
    """
    probability_metrics = _score_probability_predictions(
        y_true=y_true,
        y_proba=y_proba,
        proba_classes=proba_classes,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        **probability_metrics,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report_text": classification_report(
            y_true,
            y_pred,
            zero_division=0,
            output_dict=False,
        ),
        "classification_report_dict": classification_report(
            y_true,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }


def build_model_results_table(
    results: dict,
    primary_metric: str = "accuracy",
) -> pd.DataFrame:
    """
    Convert a results dictionary into a compact comparison table.
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "log_loss": metrics.get("log_loss", np.nan),
                "brier_score": metrics.get("brier_score", np.nan),
                "multiclass_brier": metrics.get("multiclass_brier", np.nan),
            }
        )

    sort_priority = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "brier_score",
        "multiclass_brier",
    ]
    if primary_metric not in sort_priority:
        raise ValueError(
            f"Unsupported primary metric '{primary_metric}'. "
            f"Expected one of {sort_priority}."
        )

    sort_cols = [primary_metric] + [col for col in sort_priority if col != primary_metric]
    ascending = [not metric_higher_is_better(col) for col in sort_cols]

    return pd.DataFrame(rows).sort_values(
        sort_cols,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)


def build_class_metrics_table(report_dict: dict) -> pd.DataFrame:
    """
    Convert sklearn classification_report(output_dict=True) into a class-level table.
    """
    rows = []
    for cls, metrics in report_dict.items():
        if cls in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append(
            {
                "class": cls,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": metrics["support"],
            }
        )

    return pd.DataFrame(rows)


def get_model_space(random_state: int = 42) -> dict:
    """
    Define the model set and tuning grids.

    The default setup intentionally favors models that are fast enough to be
    rerun regularly during an active season. Slower variants such as the
    classic Gradient Boosting classifier are omitted in favor of
    HistGradientBoosting, which gives a better speed/quality trade-off on this
    tabular problem.
    """
    model_space = {
        "logistic_regression": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            multi_class="multinomial",
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__C": [0.1, 1.0, 5.0],
            },
        },
        "naive_bayes": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", GaussianNB()),
                ]
            ),
            "param_grid": {
                "model__var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        },
        "knn": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            "param_grid": {
                "model__n_neighbors": [3, 5, 9, 15],
                "model__weights": ["uniform", "distance"],
            },
        },
        "svm": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVC(
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__C": [0.5, 1.0, 5.0],
                "model__kernel": ["rbf"],
                "model__gamma": ["scale", "auto"],
            },
        },
        "decision_tree": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        DecisionTreeClassifier(
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
        },
        "random_forest": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            class_weight="balanced_subsample",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__n_estimators": [200],
                "model__max_depth": [5, 8],
                "model__min_samples_leaf": [3, 5],
            },
        },
        "hist_gradient_boosting": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            early_stopping=False,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__min_samples_leaf": [20],
                "model__max_depth": [3],
            },
        },
    }

    return model_space


def get_betting_model_space(random_state: int = 42) -> dict:
    """
    Define a compact model space for probability-oriented binary pricing.
    """
    return {
        "logistic_regression": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__C": [0.3, 1.0, 3.0],
            },
        },
        "naive_bayes": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", GaussianNB()),
                ]
            ),
            "param_grid": {
                "model__var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        },
        "hist_gradient_boosting": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            early_stopping=False,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__min_samples_leaf": [20],
                "model__max_depth": [3],
            },
        },
    }


def summarize_validation_metrics(
    fold_metrics: list[dict],
    metric_names: tuple[str, ...] = (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "brier_score",
        "multiclass_brier",
    ),
) -> dict:
    """
    Average the selected validation metrics across rolling validation folds.
    """
    if len(fold_metrics) == 0:
        raise ValueError("Expected at least one validation fold.")

    summary = {}
    for metric_name in metric_names:
        values = pd.Series([metrics.get(metric_name, np.nan) for metrics in fold_metrics], dtype=float)
        summary[metric_name] = float(values.dropna().mean()) if values.notna().any() else np.nan
    return summary


def tune_models_on_validation(
    model_space: dict,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    validation_splits: list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]] | None = None,
    primary_metric: str = "accuracy",
) -> dict:
    """
    Tune all models using one or more validation splits.
    """
    supported_metrics = {
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "brier_score",
        "multiclass_brier",
    }
    if primary_metric not in supported_metrics:
        raise ValueError(
            f"Unsupported primary metric '{primary_metric}'. "
            f"Expected one of {sorted(supported_metrics)}."
        )

    if validation_splits is None:
        if any(obj is None for obj in (X_train, y_train, X_val, y_val)):
            raise ValueError(
                "Either provide X_train/y_train/X_val/y_val or validation_splits."
            )
        validation_splits = [(X_train, y_train, X_val, y_val)]

    if len(validation_splits) == 0:
        raise ValueError("Expected at least one validation split.")

    tuning_results = {}
    higher_is_better = metric_higher_is_better(primary_metric)

    for model_name, spec in model_space.items():
        best_score = -np.inf if higher_is_better else np.inf
        best_result = None

        for params in ParameterGrid(spec["param_grid"]):
            fold_metrics = []

            for X_train_fold, y_train_fold, X_val_fold, y_val_fold in validation_splits:
                estimator = clone(spec["pipeline"])
                estimator.set_params(**params)
                estimator.fit(X_train_fold, y_train_fold)

                y_val_pred = estimator.predict(X_val_fold)
                y_val_proba, proba_classes = safe_predict_proba(estimator, X_val_fold)
                fold_metrics.append(
                    score_predictions(
                        y_val_fold,
                        y_val_pred,
                        y_proba=y_val_proba,
                        proba_classes=proba_classes,
                    )
                )

            metrics = summarize_validation_metrics(fold_metrics)
            candidate_score = metrics[primary_metric]
            if pd.isna(candidate_score):
                continue

            if (
                (higher_is_better and candidate_score > best_score)
                or (not higher_is_better and candidate_score < best_score)
            ):
                best_score = candidate_score
                best_result = {
                    "best_params": params,
                    "n_validation_folds": len(validation_splits),
                    **metrics,
                }

        if best_result is None:
            raise ValueError(
                f"Unable to tune model '{model_name}' for primary metric '{primary_metric}'. "
                "No valid candidate produced the required metric."
            )

        tuning_results[model_name] = best_result

    return tuning_results


def refit_model_on_full_history(
    model_space: dict,
    model_name: str,
    best_params: dict,
    X_history: pd.DataFrame,
    y_history: pd.Series,
):
    """
    Refit one tuned model on the full historical dataset available before prediction.
    """
    estimator = clone(model_space[model_name]["pipeline"])
    estimator.set_params(**best_params)
    estimator.fit(X_history, y_history)
    return estimator


def get_tree_feature_importance(
    fitted_estimator,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Extract feature importance when available.
    """
    model = fitted_estimator.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    )

    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def get_linear_model_coefficients(
    fitted_estimator,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Extract coefficient-based interpretation for linear classifiers.

    When the pipeline uses a scaler, the coefficients describe the log-odds
    change for a one-standard-deviation increase in the corresponding feature.
    """
    model = fitted_estimator.named_steps["model"]

    if not hasattr(model, "coef_"):
        return pd.DataFrame(
            columns=[
                "feature",
                "coefficient",
                "abs_coefficient",
                "odds_ratio_per_1sd",
                "direction",
            ]
        )

    coef = np.asarray(model.coef_)
    if coef.ndim == 2:
        coef = coef[0]

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": coef,
        }
    )
    coefficient_df["abs_coefficient"] = coefficient_df["coefficient"].abs()
    coefficient_df["odds_ratio_per_1sd"] = np.exp(coefficient_df["coefficient"])
    coefficient_df["direction"] = np.where(
        coefficient_df["coefficient"] >= 0,
        "Higher value increases home-win log-odds",
        "Higher value decreases home-win log-odds",
    )

    return coefficient_df.sort_values(
        "abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)


def run_static_feature_set_screening(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    model_name: str,
    model_spec: dict,
    train_end_date: str = "2024-12-31",
    val_start_date: str = "2025-01-01",
    val_exclude_season: int = 2025,
    target_col: str = "target_1x2",
    primary_metric: str = "accuracy",
) -> pd.DataFrame:
    """
    Quickly compare multiple feature sets for one model using a static validation split.

    This is a cheap pre-selection step before the expensive rolling backtest.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    train_df = data[data["date"] <= pd.to_datetime(train_end_date)].copy()
    val_df = data[
        (data["date"] >= pd.to_datetime(val_start_date))
        & (data["season_id"] != val_exclude_season)
    ].copy()

    rows = []

    for fs_name, fs_features in feature_sets.items():
        # Keep only features actually present
        fs_features = [col for col in fs_features if col in data.columns]

        if len(fs_features) == 0:
            rows.append(
                {
                    "model": model_name,
                    "feature_set": fs_name,
                    "n_features": 0,
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                    "best_params": None,
                }
            )
            continue

        X_train = train_df[fs_features].copy()
        y_train = train_df[target_col].copy()

        X_val = val_df[fs_features].copy()
        y_val = val_df[target_col].copy()

        higher_is_better = metric_higher_is_better(primary_metric)
        best_score = -np.inf if higher_is_better else np.inf
        best_metrics = None
        best_params = None

        for params in ParameterGrid(model_spec["param_grid"]):
            estimator = clone(model_spec["pipeline"])
            estimator.set_params(**params)
            estimator.fit(X_train, y_train)

            y_val_pred = estimator.predict(X_val)
            y_val_proba, proba_classes = safe_predict_proba(estimator, X_val)
            metrics = score_predictions(
                y_val,
                y_val_pred,
                y_proba=y_val_proba,
                proba_classes=proba_classes,
            )

            if pd.isna(metrics[primary_metric]):
                continue

            if (
                (higher_is_better and metrics[primary_metric] > best_score)
                or (not higher_is_better and metrics[primary_metric] < best_score)
            ):
                best_score = metrics[primary_metric]
                best_metrics = metrics
                best_params = params

        if best_metrics is None:
            rows.append(
                {
                    "model": model_name,
                    "feature_set": fs_name,
                    "n_features": len(fs_features),
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                    "log_loss": np.nan,
                    "brier_score": np.nan,
                    "multiclass_brier": np.nan,
                    "best_params": None,
                }
            )
            continue

        rows.append(
            {
                "model": model_name,
                "feature_set": fs_name,
                "n_features": len(fs_features),
                "accuracy": best_metrics["accuracy"],
                "macro_f1": best_metrics["macro_f1"],
                "weighted_f1": best_metrics["weighted_f1"],
                "log_loss": best_metrics.get("log_loss", np.nan),
                "brier_score": best_metrics.get("brier_score", np.nan),
                "multiclass_brier": best_metrics.get("multiclass_brier", np.nan),
                "best_params": str(best_params),
            }
        )

    sort_priority = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "brier_score",
        "multiclass_brier",
    ]
    sort_cols = [primary_metric] + [col for col in sort_priority if col != primary_metric]
    ascending = [not metric_higher_is_better(col) for col in sort_cols]

    return pd.DataFrame(rows).sort_values(
        sort_cols,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)
