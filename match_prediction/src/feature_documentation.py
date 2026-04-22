import json

import pandas as pd


FEATURE_FAMILY_RULES = [
    ("benchmark_home_prob", "Market benchmark", "Implied home-win probability from pre-match market odds."),
    ("benchmark_draw_prob", "Market benchmark", "Implied draw probability from pre-match market odds."),
    ("benchmark_away_prob", "Market benchmark", "Implied away-win probability from pre-match market odds."),
    ("benchmark_home_not_lose_prob", "Market benchmark", "Implied home-not-lose probability derived from the market benchmark."),
    ("benchmark_away_not_lose_prob", "Market benchmark", "Implied away-not-lose probability derived from the market benchmark."),
    ("benchmark_home_odds", "Market benchmark", "Observed home-win odds from the selected market benchmark."),
    ("benchmark_draw_odds", "Market benchmark", "Observed draw odds from the selected market benchmark."),
    ("benchmark_away_odds", "Market benchmark", "Observed away-win odds from the selected market benchmark."),
    ("benchmark_home_not_lose_fair_odds", "Market benchmark", "Fair home-not-lose odds derived from normalized benchmark probabilities."),
    ("benchmark_away_not_lose_fair_odds", "Market benchmark", "Fair away-not-lose odds derived from normalized benchmark probabilities."),
    ("benchmark_overround", "Market benchmark", "Bookmaker margin implied by the selected 1X2 market benchmark."),
    ("elo", "Elo strength", "Pre-match Elo rating or Elo difference."),
    ("rest_days", "Rest and schedule", "Days of rest before the match."),
    ("expected_points", "Expected points", "Expected-points based form feature."),
    ("points", "Points form", "Observed points-based form feature."),
    ("goals_for", "Goals scored", "Goal production feature."),
    ("goals_against", "Goals conceded", "Goal concession feature."),
    ("goal_diff", "Goal difference", "Goal difference feature."),
    ("xg_for", "Expected goals for", "Chance creation quality feature."),
    ("xg_against", "Expected goals against", "Chance concession quality feature."),
    ("xg_diff", "Expected goal difference", "Net expected-goal feature."),
    ("np_xg_for", "Non-penalty xG for", "Open-play chance creation feature."),
    ("np_xg_against", "Non-penalty xG against", "Open-play chance concession feature."),
    ("np_xg_diff", "Non-penalty xG difference", "Net open-play chance feature."),
    ("ppda", "Pressing intensity", "PPDA-based pressing feature."),
    ("deep_completions", "Territorial progression", "Deep-completions based attacking territory feature."),
]


def infer_feature_family(feature_name: str) -> str:
    """
    Infer a broad feature family from the feature name.
    """
    for pattern, family, _ in FEATURE_FAMILY_RULES:
        if pattern in feature_name:
            return family
    return "Other pre-match feature"


def infer_feature_scope(feature_name: str) -> str:
    """
    Infer whether the feature is overall, venue-specific, cumulative, or differential.
    """
    if feature_name.startswith("diff_") or "_diff_" in feature_name:
        return "Relative difference"
    if feature_name.startswith("benchmark_"):
        return "Market-implied benchmark"
    if "_venue" in feature_name:
        return "Home/away venue split"
    if "_overall" in feature_name:
        return "All previous matches"
    if "_cum_avg_before" in feature_name:
        return "Season-to-date average"
    return "General pre-match feature"


def infer_feature_horizon(feature_name: str) -> str:
    """
    Infer the historical horizon used by the feature.
    """
    if "_avg_last_2_" in feature_name:
        return "Short-term form (last 2)"
    if "_avg_last_8_" in feature_name:
        return "Medium-term form (last 8)"
    if "_ewm_span_5_" in feature_name:
        return "Recency-weighted form (EWM span 5)"
    if "_cum_avg_before" in feature_name:
        return "Season-to-date form"
    if feature_name.startswith("benchmark_"):
        return "Pre-match market quote"
    return "Single-value pre-match feature"


def infer_feature_side(feature_name: str) -> str:
    """
    Infer whether the feature belongs to the home side, away side, or the difference.
    """
    if feature_name.startswith("home_"):
        return "Home team"
    if feature_name.startswith("away_"):
        return "Away team"
    if feature_name.startswith("diff_"):
        return "Home minus away"
    if feature_name.startswith("elo_diff_"):
        return "Home minus away"
    if feature_name.startswith("benchmark_"):
        return "Market-level"
    return "Match-level"


def infer_feature_description(feature_name: str) -> str:
    """
    Build a short human-readable feature description.
    """
    family = infer_feature_family(feature_name)
    scope = infer_feature_scope(feature_name)
    horizon = infer_feature_horizon(feature_name)
    side = infer_feature_side(feature_name)

    base_description = "Pre-match feature."
    for pattern, _, description in FEATURE_FAMILY_RULES:
        if pattern in feature_name:
            base_description = description
            break

    return f"{side}; {family}; {scope}; {horizon}. {base_description}"


def build_feature_dictionary(feature_cols: list[str]) -> pd.DataFrame:
    """
    Build a feature dictionary for thesis-style methodology documentation.
    """
    rows = []
    for feature_name in feature_cols:
        rows.append(
            {
                "feature": feature_name,
                "side": infer_feature_side(feature_name),
                "family": infer_feature_family(feature_name),
                "scope": infer_feature_scope(feature_name),
                "horizon": infer_feature_horizon(feature_name),
                "description": infer_feature_description(feature_name),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["family", "side", "feature"]
    ).reset_index(drop=True)


def summarize_feature_sets(feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    """
    Summarize feature subsets used in the modeling notebooks.
    """
    rows = []
    for feature_set_name, features in feature_sets.items():
        feature_dictionary = build_feature_dictionary(features)
        family_counts = (
            feature_dictionary["family"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        rows.append(
            {
                "feature_set": feature_set_name,
                "n_features": len(features),
                "families_present": ", ".join(sorted(feature_dictionary["family"].unique().tolist())),
                "family_counts_json": json.dumps(family_counts, ensure_ascii=False, sort_keys=True),
                "feature_list": ", ".join(features),
            }
        )

    return pd.DataFrame(rows).sort_values("n_features").reset_index(drop=True)
