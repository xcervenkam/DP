from __future__ import annotations

import os

# Prevent joblib's physical-core probe from warning in restricted Windows sessions.
logical_cores = max(1, os.cpu_count() or 1)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, min(8, logical_cores - 1))))

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.config import RANDOM_STATE


MODEL_LABELS = {
    "logistic": "Multinomial logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "svm": "Calibrated linear support vector machine",
    "knn": "k-nearest neighbours",
    "naive_bayes": "Gaussian naive Bayes",
    "gradient_boosting": "Gradient boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}

BINARY_MODEL_LABELS = {
    **MODEL_LABELS,
    "logistic": "Binary logistic regression",
}


MODEL_GRIDS = {
    "logistic": [
        {"C": 0.01},
        {"C": 0.1},
        {"C": 1.0},
    ],
    "decision_tree": [
        {"max_depth": 4, "min_samples_leaf": 25},
        {"max_depth": 6, "min_samples_leaf": 25},
        {"max_depth": 8, "min_samples_leaf": 50},
    ],
    "random_forest": [
        {"max_depth": 8, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"max_depth": 14, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"max_depth": None, "min_samples_leaf": 10, "max_features": "sqrt"},
    ],
    "svm": [
        {"C": 0.01},
        {"C": 0.1},
        {"C": 1.0},
    ],
    "knn": [
        {"n_neighbors": 25, "weights": "uniform"},
        {"n_neighbors": 50, "weights": "distance"},
        {"n_neighbors": 100, "weights": "distance"},
    ],
    "naive_bayes": [
        {"var_smoothing": 1e-8},
        {"var_smoothing": 1e-6},
        {"var_smoothing": 1e-4},
    ],
    "gradient_boosting": [
        {"n_estimators": 150, "learning_rate": 0.03, "max_depth": 2},
        {"n_estimators": 250, "learning_rate": 0.03, "max_depth": 3},
    ],
    "xgboost": [
        {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 5},
        {"n_estimators": 350, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 10},
    ],
    "lightgbm": [
        {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 30},
        {"n_estimators": 350, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50},
    ],
    "catboost": [
        {"iterations": 200, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 5},
        {"iterations": 350, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 10},
    ],
}


def make_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    """Create fold-fitted imputation, scaling and league encoding."""
    categorical = [column for column in feature_names if column == "league_id"]
    numeric = [column for column in feature_names if column not in categorical]

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def make_classifier(model_name: str, parameters: dict, task: str = "multiclass"):
    """Create one deterministic classifier for a multiclass or binary task."""
    if task not in {"multiclass", "binary"}:
        raise ValueError("task must be 'multiclass' or 'binary'.")
    if model_name == "logistic":
        return LogisticRegression(
            C=parameters["C"],
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            random_state=RANDOM_STATE,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            max_features=parameters["max_features"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_name == "svm":
        return CalibratedClassifierCV(
            estimator=LinearSVC(
                C=parameters["C"],
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
            method="sigmoid",
            cv=3,
        )
    if model_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=parameters["n_neighbors"],
            weights=parameters["weights"],
            n_jobs=-1,
        )
    if model_name == "naive_bayes":
        return GaussianNB(var_smoothing=parameters["var_smoothing"])
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=parameters["n_estimators"],
            learning_rate=parameters["learning_rate"],
            max_depth=parameters["max_depth"],
            random_state=RANDOM_STATE,
        )
    if model_name == "xgboost":
        xgb_options = {
            "objective": "multi:softprob" if task == "multiclass" else "binary:logistic",
            "n_estimators": parameters["n_estimators"],
            "learning_rate": parameters["learning_rate"],
            "max_depth": parameters["max_depth"],
            "min_child_weight": parameters["min_child_weight"],
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }
        if task == "multiclass":
            xgb_options["num_class"] = 3
        return XGBClassifier(**xgb_options)
    if model_name == "lightgbm":
        return LGBMClassifier(
            objective="multiclass" if task == "multiclass" else "binary",
            n_estimators=parameters["n_estimators"],
            learning_rate=parameters["learning_rate"],
            num_leaves=parameters["num_leaves"],
            min_child_samples=parameters["min_child_samples"],
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    if model_name == "catboost":
        return CatBoostClassifier(
            loss_function="MultiClass" if task == "multiclass" else "Logloss",
            iterations=parameters["iterations"],
            learning_rate=parameters["learning_rate"],
            depth=parameters["depth"],
            l2_leaf_reg=parameters["l2_leaf_reg"],
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=-1,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown model: {model_name}")


def ordered_probabilities(
    model,
    transformed_data,
    labels: np.ndarray,
) -> np.ndarray:
    """Return predict_proba columns in the requested H/D/A order."""
    raw_probabilities = model.predict_proba(transformed_data)
    fitted_classes = list(model.classes_)
    order = [fitted_classes.index(label) for label in labels]
    return raw_probabilities[:, order]
