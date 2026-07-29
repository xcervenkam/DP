"""Transparent fold-wise feature selection for classification notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


def _numeric_selection_matrix(
    frame: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create one imputed column per raw feature for filter-based selection."""
    columns = {}
    discrete = []
    for feature in feature_names:
        if feature == "league_id" or not pd.api.types.is_numeric_dtype(frame[feature]):
            codes, _ = pd.factorize(frame[feature].astype("string"), sort=True)
            columns[feature] = codes
            discrete.append(True)
        else:
            values = pd.to_numeric(frame[feature], errors="coerce")
            median = values.median()
            columns[feature] = values.fillna(0.0 if pd.isna(median) else median)
            discrete.append(False)
    matrix = pd.DataFrame(columns, index=frame.index)
    return matrix, np.asarray(discrete, dtype=bool)


def mutual_information_ranking(
    frame: pd.DataFrame,
    target: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """Rank raw features by I(X_j; Y), fitted on one training sample only."""
    matrix, discrete = _numeric_selection_matrix(frame, feature_names)
    scores = mutual_info_classif(
        matrix,
        target,
        discrete_features=discrete,
        random_state=RANDOM_STATE,
    )
    ranking = pd.DataFrame({"feature": feature_names, "mutual_information": scores})
    ranking = ranking.sort_values(
        ["mutual_information", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking["mi_rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def select_mi_with_correlation_control(
    frame: pd.DataFrame,
    target: pd.Series,
    feature_names: list[str],
    max_selected: int,
    mandatory_features: list[str] | None = None,
    correlation_threshold: float = 0.95,
) -> tuple[list[str], pd.DataFrame]:
    """Select high-MI features while suppressing near-duplicate numeric variables.

    Mandatory variables are always retained and do not consume the `max_selected`
    budget.  Correlations and MI are both estimated on the supplied training data.
    """
    mandatory = [
        feature
        for feature in (mandatory_features or [])
        if feature in feature_names
    ]
    candidates = [feature for feature in feature_names if feature not in mandatory]
    ranking = mutual_information_ranking(frame, target, candidates)
    numeric_candidates = [
        feature
        for feature in candidates
        if pd.api.types.is_numeric_dtype(frame[feature])
    ]
    numeric_matrix, _ = _numeric_selection_matrix(frame, numeric_candidates)
    correlation = numeric_matrix.corr(method="spearman").abs()

    selected = []
    rejected_as_redundant = []
    for feature in ranking["feature"]:
        if len(selected) >= max_selected:
            break
        if feature in numeric_candidates:
            comparable = [item for item in selected if item in numeric_candidates]
            if comparable and correlation.loc[feature, comparable].max() >= correlation_threshold:
                rejected_as_redundant.append(feature)
                continue
        selected.append(feature)

    ranking["selected"] = ranking["feature"].isin(selected)
    ranking["rejected_high_correlation"] = ranking["feature"].isin(
        rejected_as_redundant
    )
    return list(dict.fromkeys(mandatory + selected)), ranking


def summarize_selection_stability(selected_by_fold: dict[int, list[str]]) -> pd.DataFrame:
    """Count how often each raw feature is selected across temporal folds."""
    number_of_folds = len(selected_by_fold)
    counts: dict[str, int] = {}
    for features in selected_by_fold.values():
        for feature in set(features):
            counts[feature] = counts.get(feature, 0) + 1
    rows = [
        {
            "feature": feature,
            "selected_folds": count,
            "selection_frequency": count / number_of_folds,
        }
        for feature, count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["selected_folds", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def initial_feature_screen(
    frame: pd.DataFrame,
    feature_names: list[str],
    mandatory_features: list[str] | None = None,
    missing_threshold: float = 0.40,
    dominant_threshold: float = 0.995,
) -> tuple[list[str], pd.DataFrame]:
    """Remove unusable variables using one training sample only.

    The audit reports missingness, observed cardinality, dominant-value share,
    mandatory status, retention status, and the first rejection reason.
    """
    mandatory = set(mandatory_features or [])
    rows = []
    retained = []
    for feature in feature_names:
        values = frame[feature]
        missing_share = float(values.isna().mean())
        observed = values.dropna()
        unique_values = int(observed.nunique())
        dominant_share = (
            float(observed.value_counts(normalize=True, dropna=True).iloc[0])
            if len(observed)
            else 1.0
        )
        reason = "retained"
        keep = True
        if feature not in mandatory and missing_share > missing_threshold:
            keep = False
            reason = f"missing_share_above_{missing_threshold:.0%}"
        elif feature not in mandatory and unique_values <= 1:
            keep = False
            reason = "constant_or_empty"
        elif feature not in mandatory and dominant_share >= dominant_threshold:
            keep = False
            reason = f"dominant_value_at_least_{dominant_threshold:.1%}"
        elif feature in mandatory and unique_values == 0:
            keep = False
            reason = "mandatory_but_completely_missing"
        if keep:
            retained.append(feature)
        rows.append(
            {
                "feature": feature,
                "mandatory": feature in mandatory,
                "missing_share": missing_share,
                "unique_observed_values": unique_values,
                "dominant_value_share": dominant_share,
                "screen_retained": keep,
                "screen_reason": reason,
            }
        )
    return retained, pd.DataFrame(rows)


def spearman_redundancy_filter(
    frame: pd.DataFrame,
    feature_names: list[str],
    mandatory_features: list[str] | None = None,
    threshold: float = 0.92,
) -> tuple[list[str], pd.DataFrame]:
    """Greedily suppress strongly correlated numeric training variables.

    Mandatory variables are considered first and are never rejected because of
    correlation. Remaining variables are ordered by lower missingness and then
    by their original order, making the rule deterministic and transparent.
    """
    mandatory = [item for item in (mandatory_features or []) if item in feature_names]
    nonmandatory = [item for item in feature_names if item not in mandatory]
    original_order = {feature: position for position, feature in enumerate(feature_names)}
    nonmandatory.sort(key=lambda feature: (frame[feature].isna().mean(), original_order[feature]))
    ordered = mandatory + nonmandatory
    numeric = [
        feature for feature in ordered
        if feature != "league_id" and pd.api.types.is_numeric_dtype(frame[feature])
    ]
    matrix, _ = _numeric_selection_matrix(frame, numeric)
    correlation = matrix.corr(method="spearman").abs()

    retained: list[str] = []
    rows = []
    for feature in ordered:
        compared = [item for item in retained if item in numeric]
        strongest_feature = None
        strongest_correlation = np.nan
        keep = True
        reason = "retained"
        if feature in numeric and feature not in mandatory and compared:
            correlations = correlation.loc[feature, compared]
            strongest_feature = correlations.idxmax()
            strongest_correlation = float(correlations.max())
            if strongest_correlation >= threshold:
                keep = False
                reason = f"spearman_redundant_with:{strongest_feature}"
        if keep:
            retained.append(feature)
        rows.append(
            {
                "feature": feature,
                "mandatory": feature in mandatory,
                "correlation_retained": keep,
                "strongest_retained_feature": strongest_feature,
                "absolute_spearman": strongest_correlation,
                "correlation_reason": reason,
            }
        )
    return retained, pd.DataFrame(rows)


def shadow_feature_relevance(
    frame: pd.DataFrame,
    target: pd.Series,
    feature_names: list[str],
    iterations: int = 6,
    n_estimators: int = 180,
) -> tuple[list[str], pd.DataFrame]:
    """Compare real variables with independently permuted shadow variables.

    In each repetition a random forest is fitted to the real and permuted
    columns. A real variable records a hit when its importance exceeds the
    maximum shadow importance. Confirmed and tentative variables proceed to RFE.
    """
    if not feature_names:
        return [], pd.DataFrame(
            columns=["feature", "shadow_hits", "shadow_hit_rate", "mean_importance", "shadow_status"]
        )
    matrix, _ = _numeric_selection_matrix(frame, feature_names)
    matrix = matrix.astype(float)
    target_codes = pd.Categorical(target).codes
    rng = np.random.default_rng(RANDOM_STATE)
    hits = np.zeros(len(feature_names), dtype=int)
    importance_sum = np.zeros(len(feature_names), dtype=float)
    shadow_thresholds = []

    for iteration in range(iterations):
        shadow = np.column_stack(
            [rng.permutation(matrix.iloc[:, column].to_numpy()) for column in range(matrix.shape[1])]
        )
        augmented = np.column_stack([matrix.to_numpy(), shadow])
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=7,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE + iteration,
        )
        model.fit(augmented, target_codes)
        real_importance = model.feature_importances_[: len(feature_names)]
        shadow_importance = model.feature_importances_[len(feature_names) :]
        threshold = float(shadow_importance.max())
        hits += real_importance > threshold
        importance_sum += real_importance
        shadow_thresholds.append(threshold)

    hit_rate = hits / iterations
    status = np.where(hit_rate >= 0.60, "confirmed", np.where(hit_rate >= 0.34, "tentative", "rejected"))
    audit = pd.DataFrame(
        {
            "feature": feature_names,
            "shadow_hits": hits,
            "shadow_hit_rate": hit_rate,
            "mean_importance": importance_sum / iterations,
            "mean_max_shadow_importance": np.mean(shadow_thresholds),
            "shadow_status": status,
        }
    ).sort_values(["shadow_hit_rate", "mean_importance", "feature"], ascending=[False, False, True])
    retained = audit.loc[audit["shadow_status"].isin(["confirmed", "tentative"]), "feature"].tolist()
    return retained, audit.reset_index(drop=True)


def rfe_relevance_ranking(
    frame: pd.DataFrame,
    target: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """Rank a reduced numeric candidate set with logistic-regression RFE."""
    if not feature_names:
        return pd.DataFrame(columns=["feature", "rfe_rank"])
    matrix, _ = _numeric_selection_matrix(frame, feature_names)
    scaled = StandardScaler().fit_transform(matrix.astype(float))
    estimator = LogisticRegression(C=0.1, max_iter=2500, random_state=RANDOM_STATE)
    step = max(1, len(feature_names) // 10)
    selector = RFE(estimator=estimator, n_features_to_select=1, step=step)
    selector.fit(scaled, target)
    return (
        pd.DataFrame({"feature": feature_names, "rfe_rank": selector.ranking_})
        .sort_values(["rfe_rank", "feature"])
        .reset_index(drop=True)
    )
