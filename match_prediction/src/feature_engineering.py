from __future__ import annotations

import numpy as np
import pandas as pd


FIXTURE_KEY = ["league_id", "season_id", "game_id"]

BASIC_METRICS = [
    "goals_for",
    "goals_against",
    "points",
    "goal_diff",
]

RICH_METRICS = [
    "xg_for",
    "xg_against",
    "xg_diff",
    "np_xg_for",
    "np_xg_against",
    "np_xg_diff",
    "expected_points",
    "ppda",
    "deep_completions",
]

VENUE_METRICS = [
    "goals_for",
    "goals_against",
    "points",
    "xg_for",
    "xg_against",
    "expected_points",
]


def build_team_match_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert one match into one chronological row for each participating team."""
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "home_points" not in df.columns or "away_points" not in df.columns:
        df["home_points"] = np.select(
            [
                df["home_goals"] > df["away_goals"],
                df["home_goals"] == df["away_goals"],
            ],
            [3, 1],
            default=0,
        )
        df["away_points"] = np.select(
            [
                df["away_goals"] > df["home_goals"],
                df["away_goals"] == df["home_goals"],
            ],
            [3, 1],
            default=0,
        )

    common = ["league_id", "season_id", "game_id", "date"]
    home_map = {
        "home_team": "team",
        "away_team": "opponent",
        "home_goals": "goals_for",
        "away_goals": "goals_against",
        "home_xg": "xg_for",
        "away_xg": "xg_against",
        "home_np_xg": "np_xg_for",
        "away_np_xg": "np_xg_against",
        "home_expected_points": "expected_points",
        "home_ppda": "ppda",
        "home_deep_completions": "deep_completions",
        "home_points": "points",
    }
    away_map = {
        "away_team": "team",
        "home_team": "opponent",
        "away_goals": "goals_for",
        "home_goals": "goals_against",
        "away_xg": "xg_for",
        "home_xg": "xg_against",
        "away_np_xg": "np_xg_for",
        "home_np_xg": "np_xg_against",
        "away_expected_points": "expected_points",
        "away_ppda": "ppda",
        "away_deep_completions": "deep_completions",
        "away_points": "points",
    }

    home_cols = common + [column for column in home_map if column in df.columns]
    away_cols = common + [column for column in away_map if column in df.columns]
    home = df[home_cols].rename(columns=home_map).copy()
    away = df[away_cols].rename(columns=away_map).copy()
    home["is_home"] = 1
    away["is_home"] = 0

    team_matches = pd.concat([home, away], ignore_index=True)
    team_matches["goal_diff"] = team_matches["goals_for"] - team_matches["goals_against"]
    team_matches["xg_diff"] = team_matches["xg_for"] - team_matches["xg_against"]
    team_matches["np_xg_diff"] = team_matches["np_xg_for"] - team_matches["np_xg_against"]

    season_presence = (
        team_matches[["league_id", "team", "season_id"]]
        .drop_duplicates()
        .sort_values(["league_id", "team", "season_id"])
        .reset_index(drop=True)
    )
    season_group = season_presence.groupby(["league_id", "team"], sort=False)
    previous_season = season_group["season_id"].shift(1)
    season_presence["new_spell"] = (
        previous_season.isna()
        | (season_presence["season_id"].astype(int) - previous_season.astype("Int64") > 1)
    ).astype(int)
    season_presence["spell_id"] = season_presence.groupby(
        ["league_id", "team"], sort=False
    )["new_spell"].cumsum()

    season_presence = season_presence.merge(
        team_matches[["league_id", "season_id"]]
        .drop_duplicates()
        .groupby("league_id", as_index=False)["season_id"]
        .min()
        .rename(columns={"season_id": "first_data_season"}),
        on="league_id",
        how="left",
        validate="many_to_one",
    )
    season_presence["promoted_or_returning"] = season_presence["new_spell"].astype("Float64")
    season_presence.loc[
        season_presence["season_id"].eq(season_presence["first_data_season"]),
        "promoted_or_returning",
    ] = pd.NA

    team_matches = team_matches.merge(
        season_presence[
            ["league_id", "team", "season_id", "spell_id", "promoted_or_returning"]
        ],
        on=["league_id", "team", "season_id"],
        how="left",
        validate="many_to_one",
    )

    order = ["league_id", "team", "spell_id", "date", "game_id"]
    team_matches = team_matches.sort_values(order).reset_index(drop=True)
    spell_group = ["league_id", "team", "spell_id"]
    season_group = ["league_id", "team", "season_id"]

    team_matches["history_matches_before"] = team_matches.groupby(
        spell_group, sort=False
    ).cumcount()
    team_matches["matches_played_before"] = team_matches.groupby(
        season_group, sort=False
    ).cumcount()
    previous_date = team_matches.groupby(season_group, sort=False)["date"].shift(1)
    team_matches["rest_days"] = (team_matches["date"] - previous_date).dt.days
    team_matches["short_rest"] = team_matches["rest_days"].le(3).astype("Int8")
    team_matches.loc[team_matches["rest_days"].isna(), "short_rest"] = pd.NA

    return team_matches


def add_team_form_features(
    team_matches: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
    ewm_span: int = 5,
) -> pd.DataFrame:
    """Add strictly pre-match rolling, exponentially weighted and season means."""
    df = team_matches.copy()
    order = ["league_id", "team", "spell_id", "date", "game_id"]
    df = df.sort_values(order).reset_index(drop=True)
    spell_group = ["league_id", "team", "spell_id"]
    season_group = ["league_id", "team", "season_id"]
    venue_group = ["league_id", "team", "spell_id", "is_home"]
    metrics = [metric for metric in BASIC_METRICS + RICH_METRICS if metric in df.columns]

    for metric in metrics:
        for window in windows:
            df[f"{metric}_mean_last_{window}"] = df.groupby(
                spell_group, sort=False
            )[metric].transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean()
            )

        df[f"{metric}_ewm_span_{ewm_span}"] = df.groupby(
            spell_group, sort=False
        )[metric].transform(
            lambda values: values.shift(1).ewm(
                span=ewm_span,
                adjust=False,
                min_periods=1,
            ).mean()
        )
        df[f"{metric}_season_mean_pre"] = df.groupby(
            season_group, sort=False
        )[metric].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )

    for metric in [metric for metric in VENUE_METRICS if metric in df.columns]:
        df[f"{metric}_venue_mean_last_5"] = df.groupby(
            venue_group, sort=False
        )[metric].transform(
            lambda values: values.shift(1).rolling(5, min_periods=1).mean()
        )

    return df


def add_elo_features(
    matches: pd.DataFrame,
    k_factor: float = 20.0,
    home_advantage: float = 80.0,
    base_rating: float = 1500.0,
    season_regression: float = 0.75,
) -> pd.DataFrame:
    """Calculate pre-match Elo ratings separately within each league."""
    df = matches.sort_values(["date", "league_id", "game_id"]).copy()
    ratings: dict[tuple[str, str], float] = {}
    active_season: dict[str, int] = {}
    home_pre: list[float] = []
    away_pre: list[float] = []
    home_expectation: list[float] = []

    for row in df.itertuples(index=False):
        league = row.league_id
        season = int(row.season_id)
        if league in active_season and active_season[league] != season:
            for key in [key for key in ratings if key[0] == league]:
                ratings[key] = base_rating + season_regression * (
                    ratings[key] - base_rating
                )
        active_season[league] = season

        home_key = (league, row.home_team)
        away_key = (league, row.away_team)
        home_rating = ratings.get(home_key, base_rating)
        away_rating = ratings.get(away_key, base_rating)
        expected_home = 1 / (
            1 + 10 ** ((away_rating - home_rating - home_advantage) / 400)
        )

        home_pre.append(home_rating)
        away_pre.append(away_rating)
        home_expectation.append(expected_home)

        if row.home_goals > row.away_goals:
            actual_home = 1.0
        elif row.home_goals == row.away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0

        update = k_factor * (actual_home - expected_home)
        ratings[home_key] = home_rating + update
        ratings[away_key] = away_rating - update

    df["home_elo_pre"] = home_pre
    df["away_elo_pre"] = away_pre
    df["elo_diff_pre"] = df["home_elo_pre"] - df["away_elo_pre"]
    df["elo_home_expectation"] = home_expectation
    return df.sort_values(["league_id", "date", "game_id"]).reset_index(drop=True)


def add_league_context_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add league averages based only on matches played on earlier dates."""
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = (
        df.assign(total_goals=df["home_goals"] + df["away_goals"])
        .groupby(["league_id", "match_date"], as_index=False)
        .agg(
            matches=("game_id", "size"),
            home_wins=("home_win", "sum"),
            draws=("draw", "sum"),
            total_goals=("total_goals", "sum"),
        )
        .sort_values(["league_id", "match_date"])
    )

    for column in ["matches", "home_wins", "draws", "total_goals"]:
        daily[f"previous_{column}"] = daily.groupby("league_id", sort=False)[
            column
        ].transform(lambda values: values.cumsum().shift(1))

    daily["league_home_win_rate_pre"] = (
        daily["previous_home_wins"] / daily["previous_matches"]
    )
    daily["league_draw_rate_pre"] = daily["previous_draws"] / daily["previous_matches"]
    daily["league_goals_mean_pre"] = (
        daily["previous_total_goals"] / daily["previous_matches"]
    )
    context_columns = [
        "league_id",
        "match_date",
        "league_home_win_rate_pre",
        "league_draw_rate_pre",
        "league_goals_mean_pre",
    ]
    df = df.merge(
        daily[context_columns],
        on=["league_id", "match_date"],
        how="left",
        validate="many_to_one",
    )
    return df.drop(columns="match_date")


def merge_team_features(
    matches: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """Return to one row per match and add home-minus-away differences."""
    generated_patterns = (
        "_mean_last_",
        "_ewm_span_",
        "_season_mean_pre",
        "_venue_mean_last_",
    )
    general_features = [
        "history_matches_before",
        "matches_played_before",
        "rest_days",
        "short_rest",
        "promoted_or_returning",
    ]
    feature_columns = [
        column
        for column in team_features.columns
        if column in general_features
        or any(pattern in column for pattern in generated_patterns)
    ]
    keep = FIXTURE_KEY + ["is_home"] + feature_columns
    home = team_features.loc[team_features["is_home"].eq(1), keep].copy()
    away = team_features.loc[team_features["is_home"].eq(0), keep].copy()
    home = home.drop(columns="is_home").rename(
        columns={column: f"home_{column}" for column in feature_columns}
    )
    away = away.drop(columns="is_home").rename(
        columns={column: f"away_{column}" for column in feature_columns}
    )

    out = matches.merge(home, on=FIXTURE_KEY, how="left", validate="one_to_one")
    out = out.merge(away, on=FIXTURE_KEY, how="left", validate="one_to_one")

    for column in feature_columns:
        home_column = f"home_{column}"
        away_column = f"away_{column}"
        if pd.api.types.is_numeric_dtype(out[home_column]):
            out[f"diff_{column}"] = out[home_column] - out[away_column]
    return out


def build_feature_table(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full pre-match feature table and return its long intermediate table."""
    team_matches = build_team_match_table(matches)
    team_features = add_team_form_features(team_matches)
    match_features = add_elo_features(matches)
    match_features = add_league_context_features(match_features)
    match_features = merge_team_features(match_features, team_features)
    return match_features, team_features


def get_feature_sets(feature_table: pd.DataFrame) -> dict[str, list[str]]:
    """Return the Understat, structural and market information sets."""
    common = [
        "league_id",
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff_pre",
        "elo_home_expectation",
        "league_home_win_rate_pre",
        "league_draw_rate_pre",
        "league_goals_mean_pre",
        "home_history_matches_before",
        "away_history_matches_before",
        "diff_history_matches_before",
        "home_matches_played_before",
        "away_matches_played_before",
        "diff_matches_played_before",
        "home_rest_days",
        "away_rest_days",
        "diff_rest_days",
        "home_short_rest",
        "away_short_rest",
        "diff_short_rest",
        "home_promoted_or_returning",
        "away_promoted_or_returning",
        "diff_promoted_or_returning",
    ]

    clubelo_features = [
        "home_clubelo_rating",
        "away_clubelo_rating",
        "clubelo_diff",
        "clubelo_home_expectation",
        "home_competition_level",
        "away_competition_level",
        "clubelo_available",
    ]
    sofifa_features = [
        "home_sofifa_player_count",
        "away_sofifa_player_count",
        "diff_sofifa_player_count",
        "home_sofifa_top11_overall",
        "away_sofifa_top11_overall",
        "diff_sofifa_top11_overall",
        "home_sofifa_top15_overall",
        "away_sofifa_top15_overall",
        "diff_sofifa_top15_overall",
        "home_sofifa_best_gk",
        "away_sofifa_best_gk",
        "diff_sofifa_best_gk",
        "home_sofifa_top4_def",
        "away_sofifa_top4_def",
        "diff_sofifa_top4_def",
        "home_sofifa_top4_mid",
        "away_sofifa_top4_mid",
        "diff_sofifa_top4_mid",
        "home_sofifa_top3_att",
        "away_sofifa_top3_att",
        "diff_sofifa_top3_att",
        "home_sofifa_age_days",
        "away_sofifa_age_days",
        "diff_sofifa_age_days",
        "sofifa_available",
    ]
    market_features = [
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
        "market_overround",
        "market_available",
    ]

    def columns_for(metrics: list[str]) -> list[str]:
        prefixes = tuple(
            prefix
            for metric in metrics
            for prefix in (f"home_{metric}_", f"away_{metric}_", f"diff_{metric}_")
        )
        return [column for column in feature_table.columns if column.startswith(prefixes)]

    understat = [column for column in common if column in feature_table.columns]
    understat += columns_for(BASIC_METRICS + RICH_METRICS)
    structural = understat + clubelo_features + sofifa_features
    market = structural + market_features
    return {
        "understat": list(dict.fromkeys(column for column in understat if column in feature_table.columns)),
        "structural_full": list(dict.fromkeys(column for column in structural if column in feature_table.columns)),
        "market_full": list(dict.fromkeys(column for column in market if column in feature_table.columns)),
    }
