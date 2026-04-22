import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def select_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Select only clearly pre-match numeric features.
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

    return df[numeric_feature_cols].copy(), numeric_feature_cols


def train_test_split_by_date(
    df: pd.DataFrame,
    train_end_date: str,
    test_start_date: str,
    test_end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test sets based on match date.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    train_mask = data["date"] <= pd.to_datetime(train_end_date)
    test_mask = data["date"] >= pd.to_datetime(test_start_date)

    if test_end_date is not None:
        test_mask &= data["date"] <= pd.to_datetime(test_end_date)

    train_df = data.loc[train_mask].copy()
    test_df = data.loc[test_mask].copy()

    return train_df, test_df


def train_test_split_by_season(
    df: pd.DataFrame,
    train_seasons: list[int],
    test_seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test sets based on season_id.
    """
    data = df.copy()

    train_df = data[data["season_id"].isin(train_seasons)].copy()
    test_df = data[data["season_id"].isin(test_seasons)].copy()

    return train_df, test_df

def impute_with_train_medians(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Impute missing values using training-set medians.
    """
    medians = X_train.median(numeric_only=True)

    X_train_imputed = X_train.fillna(medians)
    X_test_imputed = X_test.fillna(medians)

    return X_train_imputed, X_test_imputed, medians


def fit_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Fit multinomial logistic regression baseline model.
    """
    model = LogisticRegression(
        multi_class="multinomial",
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Fit random forest baseline model.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a multiclass classifier.
    """
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=False),
        "y_pred": y_pred,
    }


def evaluate_multiple_models(results: dict) -> pd.DataFrame:
    """
    Convert a dictionary of model evaluation outputs into a compact results table.
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )


def get_random_forest_feature_importance(
    model: RandomForestClassifier,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Extract and sort feature importances from a fitted random forest model.
    """
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    )

    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
