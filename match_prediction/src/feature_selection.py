import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def select_leakage_safe_features(df: pd.DataFrame) -> list[str]:
    """
    Keep only clearly pre-match feature families.
    """
    allowed_patterns = (
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff_pre",
        "rest_days",
        "_avg_last_",
        "_ewm_span_",
        "_cum_avg_before",
        "diff_",
    )

    excluded_exact = {
        "game_id",
        "date",
        "home_team",
        "away_team",
        "target_1x2",
        "home_win",
        "draw",
        "away_win",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "season_id",
        "league_id",
    }

    feature_cols = []
    for col in df.columns:
        if col in excluded_exact:
            continue
        if any(pattern in col for pattern in allowed_patterns):
            feature_cols.append(col)

    numeric_feature_cols = (
        df[feature_cols]
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    return numeric_feature_cols


def filter_features_by_missingness(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    max_missing_share: float = 0.35,
) -> list[str]:
    """
    Keep only features with acceptable missingness based on the training set.
    """
    missing_share = train_df[feature_cols].isna().mean()
    kept = missing_share[missing_share <= max_missing_share].index.tolist()
    return kept


def filter_features_by_correlation(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.95,
) -> tuple[list[str], list[str]]:
    """
    Remove highly correlated features using training data only.
    """
    X = train_df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [col for col in feature_cols if col not in to_drop]

    return kept, to_drop


def get_permutation_selector_estimator(random_state: int = 42):
    """
    Reference model used for permutation-based feature selection.
    """
    estimator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return estimator


def permutation_select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
    estimator=None,
    top_k: int = 20,
    min_features: int = 8,
    min_importance: float = 0.0,
    n_repeats: int = 10,
    random_state: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    """
    Fit a reference model on the training set and use permutation importance
    on the validation set to select features.
    """
    if estimator is None:
        estimator = get_permutation_selector_estimator(random_state=random_state)

    estimator = clone(estimator)
    estimator.fit(X_train[feature_cols], y_train)

    result = permutation_importance(
        estimator=estimator,
        X=X_val[feature_cols],
        y=y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="f1_macro",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    selected = importance_df.loc[
        importance_df["importance_mean"] > min_importance,
        "feature",
    ].tolist()

    if top_k is not None:
        selected = selected[:top_k]

    if len(selected) < min_features:
        selected = importance_df.head(min(min_features, len(importance_df)))["feature"].tolist()

    return selected, importance_df
